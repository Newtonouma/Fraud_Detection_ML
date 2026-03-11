"""
app.py  —  AI Fraud Detector v2.0
===================================
Revised GUI — pre-trains on bundled merged_fraud_dataset.csv at startup,
then scores ANY loaded file using the trained models.

University of the West of Scotland — MSc Project
Evans Polley | B01823633
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading, os, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec

from fraud_pipeline import FraudDataPipeline, ModelManager, FEATURE_COLS

# ── Colour palette ──
C = {
    "bg": "#0D1117", "panel": "#161B22", "card": "#1C2128",
    "accent": "#2563EB", "accent2": "#7C3AED", "success": "#10B981",
    "warn": "#F59E0B", "danger": "#EF4444", "text": "#E6EDF3",
    "muted": "#8B949E", "border": "#30363D", "header_bg": "#131922",
    "gold": "#D4AC0D",
}
FF = "Segoe UI" if os.name == "nt" else "Helvetica"


class FraudDetectorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Fraud Detector v2 — UWS MSc | Evans Polley B01823633")
        self.configure(bg=C["bg"])
        try:    self.state("zoomed")
        except Exception:
            try:    self.attributes("-zoomed", True)
            except: self.geometry("1440x880")

        # Core objects
        self.train_pipeline  = FraudDataPipeline()   # holds the training data
        self.infer_pipeline  = FraudDataPipeline()   # holds any loaded file
        self.models          = ModelManager()
        self.current_file    = None
        self._df_display     = None
        self._inference_preds = {}   # {model_name: y_pred array}
        self._pretrained     = False

        self._build_ui()

        # Auto-train on startup in background
        self._start_pretrain()

    # ══════════════════════════════════════════════════════
    #  STARTUP PRE-TRAINING
    # ══════════════════════════════════════════════════════

    def _start_pretrain(self):
        def worker():
            self.after(0, lambda: self._set_status("⏳ Loading & training on merged dataset…"))
            self.after(0, self.progress_bar.start)
            self.after(0, lambda: self.progress_var.set("Pre-training models on merged dataset…"))
            try:
                self.train_pipeline.load_training_data()
                self.train_pipeline.split_and_resample(test_size=0.2, use_smote=True)
                self.models.train_all(
                    self.train_pipeline.X_train, self.train_pipeline.y_train,
                    self.train_pipeline.X_test,  self.train_pipeline.y_test,
                    progress_cb=lambda m: self.after(0, lambda msg=m: self.progress_var.set(msg))
                )
                self._pretrained = True
                # Copy scaler/imputer to infer_pipeline so it can use them
                self.infer_pipeline.scaler  = self.train_pipeline.scaler
                self.infer_pipeline.imputer = self.train_pipeline.imputer
                self.after(0, self._on_pretrain_done)
            except Exception as e:
                self.after(0, lambda: self._set_status(f"⚠️ Pre-train error: {e}"))
                self.after(0, lambda: self.progress_var.set(f"Error: {e}"))
            finally:
                self.after(0, self.progress_bar.stop)

        threading.Thread(target=worker, daemon=True).start()

    def _on_pretrain_done(self):
        valid = {k: v for k, v in self.models.results.items() if "f1" in v}
        if valid:
            best_f1  = max(v["f1"]  for v in valid.values())
            best_auc = max(v["auc"] for v in valid.values())
            self.kpi_vars["best_f1"].set(f"{best_f1:.3f}")
            self.kpi_vars["best_auc"].set(f"{best_auc:.3f}")
        # Show training metrics
        self._update_metrics_tab(source="training")
        self._update_training_kpis()
        self.progress_var.set("✅ Models pre-trained and ready!")
        self._set_status(
            "✅ Pre-trained on merged dataset (55K rows, 4 sources) — "
            "Load any CSV/XLSX/JSON to score it"
        )
        self.nb.select(0)

    def _update_training_kpis(self):
        df = self.train_pipeline.df
        if df is None: return
        n  = len(df)
        nf = int(df["is_fraud"].sum())
        self.kpi_vars["total_tx"].set(f"{n:,}")
        self.kpi_vars["fraud_tx"].set(f"{nf:,}")
        self.kpi_vars["fraud_pct"].set(f"{nf/n*100:.3f}%")
        # Draw dashboard charts from training data
        self.dash_content.pack(fill="both", expand=True)
        self.welcome_frame.pack_forget()
        self._draw_dashboard_charts(use_training=True)

    # ══════════════════════════════════════════════════════
    #  UI CONSTRUCTION
    # ══════════════════════════════════════════════════════

    def _build_ui(self):
        self._style()
        self._build_header()
        body = tk.Frame(self, bg=C["bg"])
        body.pack(fill="both", expand=True)
        self.nb = ttk.Notebook(body, style="Dark.TNotebook")
        self.nb.pack(side="left", fill="both", expand=True, padx=(8,0), pady=8)
        self._build_tab_dashboard()
        self._build_tab_table()
        self._build_tab_metrics()
        self._build_tab_charts()
        self._build_tab_mapping()
        self._build_right_panel(body)
        self._build_statusbar()

    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TFrame", background=C["bg"])
        s.configure("Dark.TNotebook", background=C["bg"], borderwidth=0)
        s.configure("Dark.TNotebook.Tab", background=C["panel"], foreground=C["muted"],
                    padding=[14,8], font=(FF,10,"bold"))
        s.map("Dark.TNotebook.Tab",
              background=[("selected",C["accent"]),("active",C["card"])],
              foreground=[("selected","white"),("active",C["text"])])
        s.configure("Treeview", background=C["card"], foreground=C["text"],
                    fieldbackground=C["card"], rowheight=26, font=(FF,9))
        s.configure("Treeview.Heading", background=C["header_bg"],
                    foreground=C["text"], font=(FF,9,"bold"))
        s.map("Treeview", background=[("selected",C["accent"])])
        s.configure("TScrollbar", background=C["panel"], troughcolor=C["bg"])

    def _build_header(self):
        hdr = tk.Frame(self, bg=C["header_bg"], height=58)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="🛡", bg=C["header_bg"], fg=C["accent"],
                 font=(FF,22)).pack(side="left", padx=(14,4))
        tk.Label(hdr, text="AI Fraud Detector", bg=C["header_bg"], fg=C["text"],
                 font=(FF,16,"bold")).pack(side="left")
        tk.Label(hdr, text=" v2.0 | UWS MSc — Evans Polley B01823633",
                 bg=C["header_bg"], fg=C["muted"], font=(FF,10)).pack(side="left")
        # Status badge
        self.pretrain_badge = tk.Label(hdr, text="⏳ Initialising…",
                                        bg=C["warn"], fg="#000",
                                        font=(FF,9,"bold"), padx=8, pady=3)
        self.pretrain_badge.pack(side="right", padx=6, pady=12)
        tk.Button(hdr, text="📂  Load File", bg=C["accent"], fg="white",
                  font=(FF,10,"bold"), relief="flat", padx=14, pady=6,
                  cursor="hand2", command=self._load_file
                  ).pack(side="right", padx=4, pady=10)

    # ── Dashboard ──
    def _build_tab_dashboard(self):
        tab = tk.Frame(self.nb, bg=C["bg"])
        self.nb.add(tab, text="  📊  Dashboard  ")

        self.welcome_frame = tk.Frame(tab, bg=C["bg"])
        self.welcome_frame.pack(expand=True, fill="both")
        tk.Label(self.welcome_frame, text="⏳", bg=C["bg"], fg=C["accent"],
                 font=(FF,52)).pack(pady=(80,10))
        tk.Label(self.welcome_frame, text="Pre-training models on merged dataset…",
                 bg=C["bg"], fg=C["text"], font=(FF,16,"bold")).pack()
        tk.Label(self.welcome_frame,
                 text="Training on 55,048 records across fraudTrain, fraudTest, "
                      "sample data & European Credit Card.\nThis takes ~30 seconds. "
                      "Load any CSV/XLSX/JSON once training is complete.",
                 bg=C["bg"], fg=C["muted"], font=(FF,11), justify="center").pack(pady=10)

        self.dash_content = tk.Frame(tab, bg=C["bg"])

        # KPI cards
        self.kpi_frame = tk.Frame(self.dash_content, bg=C["bg"])
        self.kpi_frame.pack(fill="x", padx=10, pady=(10,0))
        self.kpi_vars = {}
        for label, key, colour in [
            ("Training Rows",  "total_tx",  C["accent"]),
            ("Fraud in Train", "fraud_tx",  C["danger"]),
            ("Train Fraud %",  "fraud_pct", C["warn"]),
            ("Best F1 (train)","best_f1",   C["success"]),
            ("Best AUC-ROC",   "best_auc",  C["accent2"]),
            ("Scored Rows",    "scored",    C["gold"]),
        ]:
            card = tk.Frame(self.kpi_frame, bg=C["card"],
                            highlightbackground=colour, highlightthickness=1)
            card.pack(side="left", fill="both", expand=True, padx=4, pady=5)
            tk.Label(card, text=label, bg=C["card"], fg=C["muted"],
                     font=(FF,8)).pack(pady=(8,2))
            v = tk.StringVar(value="—"); self.kpi_vars[key] = v
            tk.Label(card, textvariable=v, bg=C["card"], fg=colour,
                     font=(FF,16,"bold")).pack(pady=(0,8))

        self.dash_fig    = plt.Figure(figsize=(14,5.5), facecolor=C["bg"])
        self.dash_canvas = FigureCanvasTkAgg(self.dash_fig, master=self.dash_content)
        self.dash_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=8)

    # ── Data Table ──
    def _build_tab_table(self):
        tab = tk.Frame(self.nb, bg=C["bg"])
        self.nb.add(tab, text="  🗄️  Data Table  ")
        toolbar = tk.Frame(tab, bg=C["panel"], height=42)
        toolbar.pack(fill="x"); toolbar.pack_propagate(False)
        tk.Label(toolbar, text="🔍", bg=C["panel"], fg=C["text"],
                 font=(FF,11)).pack(side="left", padx=(12,2), pady=10)
        self.search_var = tk.StringVar()
        self.search_var.trace("w", self._filter_table)
        tk.Entry(toolbar, textvariable=self.search_var, bg=C["card"], fg=C["text"],
                 insertbackground=C["text"], relief="flat",
                 font=(FF,9), width=32).pack(side="left", pady=8)
        for label, cmd, colour in [
            ("➕ Add",   self._crud_add,    C["success"]),
            ("✏️ Edit",  self._crud_edit,   C["accent"]),
            ("🗑️ Delete",self._crud_delete, C["danger"]),
        ]:
            tk.Button(toolbar, text=label, bg=colour, fg="white",
                      font=(FF,9,"bold"), relief="flat", padx=10, pady=2,
                      cursor="hand2", command=cmd).pack(side="left", padx=3, pady=8)
        self.row_count_var = tk.StringVar(value="Load a file to view records")
        tk.Label(toolbar, textvariable=self.row_count_var, bg=C["panel"],
                 fg=C["muted"], font=(FF,9)).pack(side="right", padx=12)
        frame = tk.Frame(tab, bg=C["bg"])
        frame.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(frame, show="headings", selectmode="browse")
        vsb = ttk.Scrollbar(frame, orient="vertical",   command=self.tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y"); hsb.pack(side="bottom", fill="x")
        self.tree.pack(fill="both", expand=True)
        self.tree.tag_configure("fraud",    background="#3B0A0A", foreground="#FCA5A5")
        self.tree.tag_configure("legit",    background=C["card"],  foreground=C["text"])
        self.tree.tag_configure("inferred", background="#0D2137", foreground="#93C5FD")
        self.tree.tag_configure("unknown",  background=C["card"],  foreground=C["muted"])

    # ── Metrics ──
    def _build_tab_metrics(self):
        tab = tk.Frame(self.nb, bg=C["bg"])
        self.nb.add(tab, text="  📈  Model Metrics  ")
        c2 = tk.Canvas(tab, bg=C["bg"], highlightthickness=0)
        vsb = ttk.Scrollbar(tab, orient="vertical", command=c2.yview)
        c2.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y"); c2.pack(fill="both", expand=True)
        self.metrics_inner = tk.Frame(c2, bg=C["bg"])
        c2.create_window((0,0), window=self.metrics_inner, anchor="nw")
        self.metrics_inner.bind("<Configure>",
            lambda e: c2.configure(scrollregion=c2.bbox("all")))
        tk.Label(self.metrics_inner,
                 text="Pre-training in progress…",
                 bg=C["bg"], fg=C["muted"], font=(FF,12)).pack(pady=60)

    # ── Charts ──
    def _build_tab_charts(self):
        tab = tk.Frame(self.nb, bg=C["bg"])
        self.nb.add(tab, text="  📉  Charts  ")
        self.charts_fig    = plt.Figure(figsize=(14,8), facecolor=C["bg"])
        self.charts_canvas = FigureCanvasTkAgg(self.charts_fig, master=tab)
        self.charts_canvas.get_tk_widget().pack(fill="both", expand=True)

    # ── Column Mapping Tab ──
    def _build_tab_mapping(self):
        tab = tk.Frame(self.nb, bg=C["bg"])
        self.nb.add(tab, text="  🗂️  Column Map  ")
        tk.Label(tab, text="Column Mapping Report",
                 bg=C["bg"], fg=C["text"],
                 font=(FF,13,"bold")).pack(pady=(16,4))
        tk.Label(tab,
                 text="Shows how your file's columns were mapped to the 52 unified features.",
                 bg=C["bg"], fg=C["muted"], font=(FF,9)).pack()
        frame = tk.Frame(tab, bg=C["bg"])
        frame.pack(fill="both", expand=True, padx=10, pady=8)
        self.map_tree = ttk.Treeview(frame, columns=("original","mapped_to","method"),
                                      show="headings", selectmode="none")
        for col, w, label in [("original",240,"Original Column"),
                               ("mapped_to",240,"Mapped Feature"),
                               ("method",200,"Method")]:
            self.map_tree.heading(col, text=label)
            self.map_tree.column(col, width=w, anchor="w")
        vsb2 = ttk.Scrollbar(frame, orient="vertical", command=self.map_tree.yview)
        self.map_tree.configure(yscrollcommand=vsb2.set)
        vsb2.pack(side="right", fill="y")
        self.map_tree.pack(fill="both", expand=True)
        self.map_tree.tag_configure("direct",  foreground=C["success"])
        self.map_tree.tag_configure("derived", foreground=C["warn"])
        self.map_tree.tag_configure("default", foreground=C["muted"])

    # ── Right Panel ──
    def _build_right_panel(self, parent):
        panel = tk.Frame(parent, bg=C["panel"], width=278)
        panel.pack(side="right", fill="y", padx=(0,8), pady=8)
        panel.pack_propagate(False)

        def section(t):
            tk.Label(panel, text=t, bg=C["panel"], fg=C["accent"],
                     font=(FF,10,"bold")).pack(anchor="w", padx=14, pady=(14,4))
            tk.Frame(panel, bg=C["border"], height=1).pack(fill="x", padx=12, pady=(0,8))

        section("FILE OPERATIONS")
        self._rb(panel, "📂  Load Any File (CSV/XLSX/JSON)",
                 self._load_file, C["accent"])
        self._rb(panel, "🔄  Reload Current File",
                 self._reload_file, C["card"])
        self._rb(panel, "💾  Export Scored Results",
                 self._export_results, C["card"])

        section("SCORING")
        tk.Label(panel, text="Score with model:", bg=C["panel"],
                 fg=C["muted"], font=(FF,9)).pack(anchor="w", padx=14)
        self.score_model_var = tk.StringVar(value="Best Model (Auto)")
        ttk.Combobox(panel, textvariable=self.score_model_var,
                     values=["Best Model (Auto)"] + list(ModelManager.MODELS.keys()),
                     state="readonly", font=(FF,9)
                     ).pack(fill="x", padx=12, pady=4)
        self._rb(panel, "▶  Score Loaded File", self._run_inference, C["success"])

        section("RE-TRAIN (OPTIONAL)")
        tk.Label(panel, text="Retrain model:", bg=C["panel"],
                 fg=C["muted"], font=(FF,9)).pack(anchor="w", padx=14)
        self.model_var = tk.StringVar(value="All Models")
        ttk.Combobox(panel, textvariable=self.model_var,
                     values=["All Models"] + list(ModelManager.MODELS.keys()),
                     state="readonly", font=(FF,9)
                     ).pack(fill="x", padx=12, pady=4)
        self.test_size_var = tk.DoubleVar(value=0.2)
        tk.Scale(panel, from_=0.1, to=0.4, resolution=0.05, orient="horizontal",
                 variable=self.test_size_var, bg=C["panel"], fg=C["text"],
                 troughcolor=C["card"], highlightthickness=0, font=(FF,8)
                 ).pack(fill="x", padx=12)
        self.smote_var = tk.BooleanVar(value=True)
        tk.Checkbutton(panel, text="Use SMOTE", variable=self.smote_var,
                       bg=C["panel"], fg=C["text"], selectcolor=C["accent"],
                       activebackground=C["panel"], font=(FF,9)
                       ).pack(anchor="w", padx=12, pady=2)
        self._rb(panel, "🔁  Re-Train on Loaded File",
                 self._run_retrain, C["accent2"])

        section("CHART OPTIONS")
        self.chart_type_var = tk.StringVar(value="All Charts")
        ttk.Combobox(panel, textvariable=self.chart_type_var,
                     values=["All Charts","Line Graph Only",
                             "Donut Chart Only","Confusion Matrix"],
                     state="readonly", font=(FF,9)
                     ).pack(fill="x", padx=12, pady=4)
        self._rb(panel, "🔃  Refresh Charts", self._refresh_charts, C["card"])

        section("DATASET INFO")
        self.info_vars = {}
        for label, key in [("File","file"),("Rows","rows"),("Cols","cols"),
                            ("Fraud","fraud"),("Mode","mode")]:
            row = tk.Frame(panel, bg=C["panel"]); row.pack(fill="x", padx=14, pady=1)
            tk.Label(row, text=label+":", bg=C["panel"], fg=C["muted"],
                     font=(FF,9), width=7, anchor="w").pack(side="left")
            v = tk.StringVar(value="—"); self.info_vars[key] = v
            tk.Label(row, textvariable=v, bg=C["panel"], fg=C["text"],
                     font=(FF,9,"bold"), wraplength=160, justify="left"
                     ).pack(side="left")

        section("PROGRESS")
        self.progress_var = tk.StringVar(value="Initialising…")
        tk.Label(panel, textvariable=self.progress_var, bg=C["panel"],
                 fg=C["muted"], font=(FF,9), wraplength=248
                 ).pack(padx=12)
        self.progress_bar = ttk.Progressbar(panel, mode="indeterminate")
        self.progress_bar.pack(fill="x", padx=12, pady=8)

        section("ABOUT")
        tk.Label(panel,
                 text="UWS MSc Computer Science\n"
                      "Evans Polley | B01823633\n\n"
                      "v2: Pre-trained on 55K rows\n"
                      "Universal column mapper\n"
                      "Inference on any file format",
                 bg=C["panel"], fg=C["muted"],
                 font=(FF,8), justify="left"
                 ).pack(anchor="w", padx=14, pady=(0,14))

    def _rb(self, parent, text, cmd, colour):
        tk.Button(parent, text=text, bg=colour, fg="white",
                  font=(FF,9,"bold"), relief="flat", padx=10, pady=6,
                  cursor="hand2", command=cmd).pack(fill="x", padx=12, pady=3)

    def _build_statusbar(self):
        sb = tk.Frame(self, bg=C["header_bg"], height=24)
        sb.pack(fill="x", side="bottom"); sb.pack_propagate(False)
        self.status_var = tk.StringVar(value="Initialising…")
        tk.Label(sb, textvariable=self.status_var, bg=C["header_bg"],
                 fg=C["muted"], font=(FF,8)).pack(side="left", padx=10)
        # Model readiness indicator
        self.model_status_var = tk.StringVar(value="⏳ Training")
        tk.Label(sb, textvariable=self.model_status_var, bg=C["header_bg"],
                 fg=C["warn"], font=(FF,8,"bold")).pack(side="right", padx=10)

    # ══════════════════════════════════════════════════════
    #  FILE LOADING
    # ══════════════════════════════════════════════════════

    def _load_file(self):
        path = filedialog.askopenfilename(
            title="Open Dataset — Any Column Format Supported",
            filetypes=[("Supported","*.csv *.xlsx *.xls *.json"),
                       ("CSV","*.csv"),("Excel","*.xlsx *.xls"),
                       ("JSON","*.json"),("All","*.*")])
        if not path: return
        self.current_file = path
        self._do_load(path)

    def _reload_file(self):
        if self.current_file: self._do_load(self.current_file)
        else: messagebox.showinfo("No File","Load a file first.")

    def _do_load(self, path):
        def worker():
            self._set_status(f"Loading {os.path.basename(path)}…")
            self.after(0, self.progress_bar.start)
            self.after(0, lambda: self.progress_var.set("Mapping columns…"))
            try:
                df = self.infer_pipeline.load_external(path)
                self.after(0, lambda: self._on_file_loaded(df, path))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Load Error", str(e)))
            finally:
                self.after(0, self.progress_bar.stop)
        threading.Thread(target=worker, daemon=True).start()

    def _on_file_loaded(self, df, path):
        raw = self.infer_pipeline.raw_df
        n   = len(raw)
        nf  = int((df["is_fraud"] == 1).sum()) if self.infer_pipeline.has_labels else 0

        self.info_vars["file"].set(os.path.basename(path)[:20])
        self.info_vars["rows"].set(f"{n:,}")
        self.info_vars["cols"].set(str(len(raw.columns)))
        self.info_vars["fraud"].set(f"{nf:,}" if self.infer_pipeline.has_labels else "unknown")
        self.info_vars["mode"].set("Labelled" if self.infer_pipeline.has_labels else "Inference")

        # Show original columns in the table
        self._populate_table(raw, df)
        self._update_mapping_tab()
        self._draw_dashboard_charts(use_training=False)

        self.progress_var.set(
            "✅ File loaded. Click '▶ Score Loaded File' to run fraud detection."
        )
        self._set_status(
            f"Loaded {n:,} rows from {os.path.basename(path)} — "
            f"{len(raw.columns)} original columns mapped to 52 features"
        )

        # Auto-score if models are ready
        if self._pretrained:
            self._run_inference()

    # ══════════════════════════════════════════════════════
    #  INFERENCE (scoring loaded file with pre-trained models)
    # ══════════════════════════════════════════════════════

    def _run_inference(self):
        if self.infer_pipeline.df is None:
            messagebox.showinfo("No File","Load a file first."); return
        if not self._pretrained:
            messagebox.showinfo("Not Ready",
                "Models are still training. Please wait a moment."); return

        def worker():
            self.after(0, self.progress_bar.start)
            self.after(0, lambda: self.progress_var.set("Scoring transactions…"))
            try:
                X = self.infer_pipeline.get_inference_X()
                sel = self.score_model_var.get()
                def cb(m): self.after(0, lambda msg=m: self.progress_var.set(msg))

                if sel == "Best Model (Auto)":
                    preds = self.models.predict_all(X, progress_cb=cb)
                else:
                    if sel not in self.models.trained:
                        raise ValueError(f"Model '{sel}' not trained.")
                    m = self.models.trained[sel]
                    y_pred = m.predict(X)
                    try:   y_prob = m.predict_proba(X)[:,1]
                    except: y_prob = y_pred.astype(float)
                    preds = {sel: {"predictions": y_pred, "probabilities": y_prob}}

                self._inference_preds = preds
                self.after(0, lambda: self._on_inference_done(preds))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Scoring Error", str(e)))
            finally:
                self.after(0, self.progress_bar.stop)
        threading.Thread(target=worker, daemon=True).start()

    def _on_inference_done(self, preds):
        # Pick best model for display
        valid = {k: v for k, v in self.models.results.items() if "f1" in v}
        best  = max(valid, key=lambda k: valid[k]["f1"]) if valid else list(preds.keys())[0]

        if best in preds and "predictions" in preds[best]:
            y_pred = preds[best]["predictions"]
            y_prob = preds[best].get("probabilities", y_pred.astype(float))
            n_fraud = int(y_pred.sum())
            n_total = len(y_pred)

            self.kpi_vars["scored"].set(f"{n_fraud:,} / {n_total:,}")
            self.info_vars["fraud"].set(f"{n_fraud:,} detected")

            # Add prediction columns to the raw display
            self._update_table_with_preds(y_pred, y_prob)
            self._draw_charts(y_pred=y_pred, y_prob=y_prob)

            # If we have true labels, compute metrics
            y_true = self.infer_pipeline.df["is_fraud"].values
            labelled = y_true >= 0
            if labelled.sum() > 0 and labelled.sum() == len(y_pred):
                self._compute_and_show_inference_metrics(y_true[labelled].astype(int),
                                                          y_pred[labelled])

        self.progress_var.set(f"✅ Scored {len(y_pred):,} rows — {n_fraud:,} fraud detected")
        self._set_status(
            f"Inference complete — {n_fraud:,} fraud detected by {best}"
        )
        self.nb.select(3)  # go to charts

    def _compute_and_show_inference_metrics(self, y_true, y_pred):
        """Show metrics when the loaded file has true labels."""
        # Add inference results to models.results under an "Inference" key
        best_valid = {k: v for k, v in self.models.results.items() if "f1" in v}
        if best_valid:
            best = max(best_valid, key=lambda k: best_valid[k]["f1"])
            try:
                auc = roc_auc_score(y_true,
                    self.models.trained[best].predict_proba(
                        self.infer_pipeline.get_inference_X())[:,1])
            except Exception:
                auc = 0.0
            from sklearn.metrics import (accuracy_score, precision_score,
                                          recall_score, f1_score, confusion_matrix)
            self.models.results["★ Inference Result"] = {
                "accuracy":  accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall":    recall_score(y_true, y_pred, zero_division=0),
                "f1":        f1_score(y_true, y_pred, zero_division=0),
                "auc":       auc,
                "cm":        confusion_matrix(y_true, y_pred),
                "y_pred":    y_pred,
            }
        self._update_metrics_tab(source="inference")
        self.nb.select(2)

    # ══════════════════════════════════════════════════════
    #  RE-TRAIN ON LOADED FILE
    # ══════════════════════════════════════════════════════

    def _run_retrain(self):
        if self.infer_pipeline.df is None:
            messagebox.showinfo("No File","Load a labelled file first."); return
        if not self.infer_pipeline.has_labels:
            messagebox.showwarning("No Labels",
                "The loaded file has no fraud label column.\n"
                "Re-training requires a labelled dataset."); return

        def worker():
            self.after(0, self.progress_bar.start)
            try:
                cb = lambda m: self.after(0, lambda msg=m: self.progress_var.set(msg))
                cb("Splitting & resampling…")
                # Retrain using the infer_pipeline data
                self.infer_pipeline.split_and_resample(
                    test_size=self.test_size_var.get(),
                    use_smote=self.smote_var.get())
                sel = self.model_var.get()
                if sel == "All Models":
                    self.models.train_all(
                        self.infer_pipeline.X_train, self.infer_pipeline.y_train,
                        self.infer_pipeline.X_test,  self.infer_pipeline.y_test,
                        progress_cb=cb)
                else:
                    cb(f"Training {sel}…")
                    self.models.train_single(sel,
                        self.infer_pipeline.X_train, self.infer_pipeline.y_train,
                        self.infer_pipeline.X_test,  self.infer_pipeline.y_test)
                # Update scaler/imputer reference
                self.infer_pipeline.scaler  = self.infer_pipeline.scaler
                self.infer_pipeline.imputer = self.infer_pipeline.imputer
                self.after(0, self._on_retrain_done)
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Retrain Error", str(e)))
            finally:
                self.after(0, self.progress_bar.stop)
        threading.Thread(target=worker, daemon=True).start()

    def _on_retrain_done(self):
        valid = {k: v for k, v in self.models.results.items() if "f1" in v}
        if valid:
            self.kpi_vars["best_f1"].set(f"{max(v['f1'] for v in valid.values()):.3f}")
            self.kpi_vars["best_auc"].set(f"{max(v['auc'] for v in valid.values()):.3f}")
        self._update_metrics_tab(source="training")
        self._draw_charts()
        self.progress_var.set("✅ Re-training complete!")
        self._set_status("Models re-trained on loaded file")
        self.nb.select(2)

    # ══════════════════════════════════════════════════════
    #  TABLE
    # ══════════════════════════════════════════════════════

    def _populate_table(self, raw_df, mapped_df=None):
        self.tree.delete(*self.tree.get_children())
        # Show original columns + append fraud_predicted if available
        show = list(raw_df.columns[:12])
        if "is_fraud" not in show and mapped_df is not None and "is_fraud" in mapped_df.columns:
            show = show  # will add pred col later
        self.tree["columns"] = show
        for col in show:
            self.tree.heading(col, text=col, command=lambda c=col: self._sort_by(c, raw_df))
            self.tree.column(col, width=max(80, min(200, len(col)*11)), anchor="w")

        self._df_display = raw_df[show].copy().reset_index(drop=True)
        self._df_mapped  = mapped_df
        self._load_tree_rows(self._df_display, mapped_df)

    def _load_tree_rows(self, df, mapped_df=None):
        self.tree.delete(*self.tree.get_children())
        for i, row in df.iterrows():
            vals = []
            for v in row:
                if isinstance(v, float): vals.append(f"{v:.3f}" if abs(v)<1e5 else f"{v:.2e}")
                else: vals.append(str(v)[:55])

            # Determine tag
            tag = "unknown"
            if mapped_df is not None and i < len(mapped_df):
                lbl = mapped_df.at[i, "is_fraud"] if "is_fraud" in mapped_df.columns else -1
                if hasattr(self, "_pred_col") and i < len(self._pred_col):
                    tag = "fraud" if self._pred_col[i] == 1 else "inferred"
                elif lbl == 1:
                    tag = "fraud"
                elif lbl == 0:
                    tag = "legit"
            self.tree.insert("", "end", iid=str(i), values=vals, tags=(tag,))

        nf = sum(1 for t in self.tree.tag_has("fraud")) if hasattr(self.tree, "tag_has") else 0
        self.row_count_var.set(f"{len(df):,} rows loaded")

    def _update_table_with_preds(self, y_pred, y_prob):
        self._pred_col = y_pred
        if self._df_display is None: return
        df = self._df_display.copy()

        # Add prediction columns
        df["🤖 Fraud_Predicted"] = ["🔴 FRAUD" if p == 1 else "✅ Legit" for p in y_pred]
        df["📊 Confidence"]      = [f"{p*100:.1f}%" for p in y_prob]

        show = list(df.columns)
        self.tree["columns"] = show
        for col in show:
            self.tree.heading(col, text=col)
            w = 120 if col.startswith("🤖") or col.startswith("📊") else max(80, min(200, len(col)*11))
            self.tree.column(col, width=w, anchor="w")

        self.tree.delete(*self.tree.get_children())
        for i, row in df.iterrows():
            vals = [str(v)[:55] for v in row]
            tag  = "fraud" if y_pred[i] == 1 else "inferred"
            self.tree.insert("", "end", iid=str(i), values=vals, tags=(tag,))

        nf = int(y_pred.sum())
        self.row_count_var.set(
            f"{len(df):,} rows  |  🔴 {nf:,} fraud  |  "
            f"✅ {len(df)-nf:,} legitimate"
        )

    def _filter_table(self, *_):
        if self._df_display is None: return
        q = self.search_var.get().lower()
        if not q:
            self._load_tree_rows(self._df_display, self._df_mapped); return
        mask = self._df_display.astype(str).apply(
            lambda col: col.str.lower().str.contains(q, na=False)).any(axis=1)
        self._load_tree_rows(self._df_display[mask])

    def _sort_by(self, col, df=None):
        if self._df_display is None: return
        asc = not getattr(self, f"_s_{col}", True)
        setattr(self, f"_s_{col}", asc)
        self._df_display = self._df_display.sort_values(
            col, ascending=asc).reset_index(drop=True)
        self._load_tree_rows(self._df_display)

    def _get_idx(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("No Selection","Select a row."); return None
        return int(sel[0])

    def _crud_add(self):
        if self._df_display is None:
            messagebox.showinfo("No Data","Load a file first."); return
        self._row_editor("add")

    def _crud_edit(self):
        idx = self._get_idx()
        if idx is not None: self._row_editor("edit", idx)

    def _crud_delete(self):
        idx = self._get_idx()
        if idx is not None and messagebox.askyesno("Delete","Delete this record?"):
            self._df_display = self._df_display.drop(index=idx).reset_index(drop=True)
            self._load_tree_rows(self._df_display)

    def _row_editor(self, mode="add", idx=None):
        win = tk.Toplevel(self)
        win.title("Add Record" if mode=="add" else "Edit Record")
        win.configure(bg=C["bg"]); win.geometry("560x540"); win.grab_set()
        canvas = tk.Canvas(win, bg=C["bg"], highlightthickness=0)
        vsb = ttk.Scrollbar(win, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y"); canvas.pack(fill="both", expand=True)
        inner = tk.Frame(canvas, bg=C["bg"])
        canvas.create_window((0,0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        cols = list(self._df_display.columns)
        entries = {}
        for i, col in enumerate(cols):
            tk.Label(inner, text=col, bg=C["bg"], fg=C["muted"],
                     font=(FF,9), anchor="w", width=28
                     ).grid(row=i, column=0, sticky="w", padx=16, pady=2)
            v = tk.StringVar()
            if mode == "edit" and idx is not None:
                v.set(str(self._df_display.at[idx, col]))
            tk.Entry(inner, textvariable=v, bg=C["card"], fg=C["text"],
                     insertbackground=C["text"], relief="flat",
                     font=(FF,9), width=38).grid(row=i, column=1, padx=10, pady=2)
            entries[col] = v
        def save():
            nr = {c: v.get() for c, v in entries.items()}
            if mode == "add":
                self._df_display = pd.concat(
                    [self._df_display, pd.DataFrame([nr])], ignore_index=True)
            else:
                for c, v in entries.items():
                    self._df_display.at[idx, c] = v.get()
            self._load_tree_rows(self._df_display); win.destroy()
        tk.Button(inner, text="💾  Save", bg=C["success"], fg="white",
                  font=(FF,10,"bold"), relief="flat", padx=20, pady=6,
                  command=save).grid(row=len(cols), column=0, columnspan=2, pady=16)

    # ══════════════════════════════════════════════════════
    #  COLUMN MAPPING TAB
    # ══════════════════════════════════════════════════════

    def _update_mapping_tab(self):
        self.map_tree.delete(*self.map_tree.get_children())
        if self.infer_pipeline.df is None: return
        raw   = self.infer_pipeline.raw_df
        mapped = self.infer_pipeline.df
        if raw is None: return

        raw_cols = [c.strip().lower().replace(" ","_").replace("-","_")
                    for c in raw.columns]

        for feat in FEATURE_COLS:
            # Check if it was directly found or derived
            if feat in raw_cols:
                orig = raw.columns[raw_cols.index(feat)]
                self.map_tree.insert("", "end",
                    values=(orig, feat, "✅ Direct match"),
                    tags=("direct",))
            elif feat.startswith("cat_"):
                self.map_tree.insert("", "end",
                    values=("(category col)", feat, "🔶 Derived from category"),
                    tags=("derived",))
            elif feat in ["trans_hour","trans_dow","trans_month","trans_day"]:
                self.map_tree.insert("", "end",
                    values=("(datetime col)", feat, "🔶 Derived from datetime"),
                    tags=("derived",))
            elif feat == "age":
                self.map_tree.insert("", "end",
                    values=("(dob col)", feat, "🔶 Derived from date of birth"),
                    tags=("derived",))
            elif feat == "geo_distance":
                self.map_tree.insert("", "end",
                    values=("(lat/long cols)", feat, "🔶 Derived from coordinates"),
                    tags=("derived",))
            elif feat == "amt_zscore":
                self.map_tree.insert("", "end",
                    values=("(amount col)", feat, "🔶 Computed from amount"),
                    tags=("derived",))
            else:
                # Check keyword match
                from fraud_pipeline import COLUMN_KEYWORD_MAP, _find_col
                found = _find_col(raw, COLUMN_KEYWORD_MAP.get(feat, [feat]))
                if found:
                    self.map_tree.insert("", "end",
                        values=(found, feat, "✅ Keyword match"),
                        tags=("direct",))
                else:
                    self.map_tree.insert("", "end",
                        values=("—", feat, "⬜ Defaulted to 0"),
                        tags=("default",))

    # ══════════════════════════════════════════════════════
    #  METRICS TAB
    # ══════════════════════════════════════════════════════

    def _update_metrics_tab(self, source="training"):
        for w in self.metrics_inner.winfo_children(): w.destroy()
        results = self.models.results
        if not results:
            tk.Label(self.metrics_inner, text="No results yet.",
                     bg=C["bg"], fg=C["muted"], font=(FF,12)).pack(pady=40); return

        title = "Training Performance" if source=="training" else "Inference Performance"
        tk.Label(self.metrics_inner, text=title,
                 bg=C["bg"], fg=C["text"], font=(FF,14,"bold")).pack(pady=(16,4))

        if source == "training":
            tk.Label(self.metrics_inner,
                     text=f"Trained on {len(self.train_pipeline.y_train):,} rows  |  "
                          f"Tested on {len(self.train_pipeline.y_test):,} rows  |  "
                          f"SMOTE: Yes  |  4 sources merged",
                     bg=C["bg"], fg=C["muted"], font=(FF,9)).pack(pady=(0,12))
        else:
            tk.Label(self.metrics_inner,
                     text="Evaluated on loaded file (true labels used where available)",
                     bg=C["bg"], fg=C["muted"], font=(FF,9)).pack(pady=(0,12))

        metrics = [("Accuracy","accuracy",C["accent"]),
                   ("Precision","precision",C["warn"]),
                   ("Recall","recall",C["success"]),
                   ("F1-Score","f1",C["accent2"]),
                   ("AUC-ROC","auc",C["danger"])]

        for mname, res in results.items():
            if "error" in res:
                card = tk.Frame(self.metrics_inner, bg=C["card"],
                                highlightbackground=C["danger"], highlightthickness=1)
                card.pack(fill="x", padx=16, pady=6)
                tk.Label(card, text=f"{mname}: {res['error']}", bg=C["card"],
                         fg=C["danger"], font=(FF,9)).pack(padx=12, pady=8)
                continue

            hl = C["gold"] if mname.startswith("★") else C["border"]
            card = tk.Frame(self.metrics_inner, bg=C["card"],
                            highlightbackground=hl, highlightthickness=2 if mname.startswith("★") else 1)
            card.pack(fill="x", padx=16, pady=8)
            hdr = tk.Frame(card, bg=C["header_bg"]); hdr.pack(fill="x")
            tk.Label(hdr, text=f"  {mname}", bg=C["header_bg"], fg=C["text"],
                     font=(FF,11,"bold")).pack(side="left", pady=8)
            cm = res.get("cm")
            if cm is not None and cm.shape == (2,2):
                tn,fp,fn,tp = cm.ravel()
                tk.Label(hdr, text=f"TP:{tp}  FP:{fp}  TN:{tn}  FN:{fn}",
                         bg=C["header_bg"], fg=C["muted"],
                         font=(FF,9)).pack(side="right", padx=12)
            row = tk.Frame(card, bg=C["card"]); row.pack(fill="x", padx=12, pady=10)
            for label, key, colour in metrics:
                val = res.get(key, 0)
                cf  = tk.Frame(row, bg=C["card"]); cf.pack(side="left", expand=True)
                tk.Label(cf, text=label, bg=C["card"], fg=C["muted"], font=(FF,8)).pack()
                tk.Label(cf, text=f"{val:.3f}", bg=C["card"], fg=colour,
                         font=(FF,15,"bold")).pack()
                bb = tk.Frame(cf, bg=C["border"], height=5, width=80)
                bb.pack(pady=(2,0)); bb.pack_propagate(False)
                tk.Frame(bb, bg=colour, height=5,
                         width=max(2,int(val*80))).place(x=0, y=0)

    # ══════════════════════════════════════════════════════
    #  CHARTS
    # ══════════════════════════════════════════════════════

    def _draw_dashboard_charts(self, use_training=False):
        df = self.train_pipeline.df if use_training else self.infer_pipeline.df
        if df is None: return
        self.dash_fig.clear(); self.dash_fig.patch.set_facecolor(C["bg"])
        gs = GridSpec(1, 3, figure=self.dash_fig, wspace=0.4)

        ax0 = self.dash_fig.add_subplot(gs[0]); ax0.set_facecolor(C["card"])
        self._draw_line_chart(ax0, df)

        ax1 = self.dash_fig.add_subplot(gs[1]); ax1.set_facecolor(C["bg"])
        self._draw_donut(ax1, df)
        ax1.set_title("Transactions vs Fraud", color=C["text"], fontsize=9, pad=6)

        ax2 = self.dash_fig.add_subplot(gs[2]); ax2.set_facecolor(C["card"])
        self._draw_source_bar(ax2, df)

        self.dash_canvas.draw()

    def _draw_charts(self, y_pred=None, y_prob=None):
        self.charts_fig.clear(); self.charts_fig.patch.set_facecolor(C["bg"])
        df      = self.infer_pipeline.df if self.infer_pipeline.df is not None else self.train_pipeline.df
        results = self.models.results
        ct      = self.chart_type_var.get()

        if ct == "Line Graph Only":
            ax = self.charts_fig.add_subplot(111); self._draw_line_chart(ax, df)
        elif ct == "Donut Chart Only":
            ax = self.charts_fig.add_subplot(111)
            self._draw_donut(ax, df, y_pred=y_pred)
            ax.set_title("Transactions vs Fraud (Twin Donut)", color=C["text"], fontsize=11)
        elif ct == "Confusion Matrix":
            valid = {k:v for k,v in results.items() if "cm" in v}
            n = len(valid)
            if n == 0: return
            for i,(mname,res) in enumerate(valid.items()):
                ax = self.charts_fig.add_subplot((n+1)//2, min(n,2), i+1)
                self._draw_cm(ax, res["cm"], mname)
        else:
            gs = GridSpec(2, 2, figure=self.charts_fig, hspace=0.45, wspace=0.4)
            ax0 = self.charts_fig.add_subplot(gs[0,0]); ax0.set_facecolor(C["card"])
            self._draw_line_chart(ax0, df, y_pred=y_pred)
            ax1 = self.charts_fig.add_subplot(gs[0,1]); ax1.set_facecolor(C["bg"])
            self._draw_donut(ax1, df, y_pred=y_pred)
            ax1.set_title("Transactions vs Fraud (Twin Donut)", color=C["text"], fontsize=9)
            ax2 = self.charts_fig.add_subplot(gs[1,0]); ax2.set_facecolor(C["card"])
            self._draw_model_bar(ax2, results)
            ax3 = self.charts_fig.add_subplot(gs[1,1]); ax3.set_facecolor(C["card"])
            valid = {k:v for k,v in results.items() if "cm" in v}
            if valid:
                best = max(valid, key=lambda k: valid[k].get("f1",0))
                self._draw_cm(ax3, valid[best]["cm"], f"Best Model: {best}")
        self.charts_canvas.draw()

    def _draw_line_chart(self, ax, df, y_pred=None):
        ax.set_facecolor(C["card"])
        if "trans_datetime" in df.columns and df["trans_datetime"].notna().any():
            tmp = df.copy()
            tmp["date"] = pd.to_datetime(tmp["trans_datetime"], utc=True, errors="coerce").dt.date

            if y_pred is not None:
                tmp["_pred"] = y_pred
                daily_pred  = tmp.groupby("date")["_pred"].sum()
                daily_total = tmp.groupby("date").size()
                x = range(len(daily_total))
                ax.plot(x, daily_total.values, color=C["accent"], lw=1.5, label="All")
                ax.plot(x, daily_pred.values,  color=C["danger"],  lw=2,
                        label="Fraud (predicted)", marker="o", markersize=3)
                ax.fill_between(x, daily_pred.values, alpha=0.35, color=C["danger"])
            else:
                daily = tmp.groupby(["date","is_fraud"]).size().unstack(fill_value=0)
                x = range(len(daily))
                if 0 in daily.columns:
                    ax.plot(x, daily[0], color=C["accent"], lw=1.5, label="Legitimate")
                    ax.fill_between(x, daily[0], alpha=0.12, color=C["accent"])
                if 1 in daily.columns:
                    ax.plot(x, daily[1], color=C["danger"], lw=2,
                            label="Fraud", marker="o", markersize=3)
                    ax.fill_between(x, daily[1], alpha=0.35, color=C["danger"])
                daily_ref = daily
                x = range(len(daily_ref))
            step = max(1, (len(daily_total) if y_pred is not None else len(daily))//7)
            ref  = daily_total if y_pred is not None else daily
            ax.set_xticks(list(range(0, len(ref), step)))
            ax.set_xticklabels([str(d) for d in ref.index[::step]],
                                rotation=30, fontsize=6, color=C["muted"])
        else:
            ax.text(0.5, 0.5, "No datetime column in this file",
                    ha="center", va="center", color=C["muted"], transform=ax.transAxes)
        ax.set_title("Fraud Over Time (Line)", color=C["text"], fontsize=9, pad=6)
        ax.tick_params(colors=C["muted"], labelsize=7)
        for sp in ax.spines.values(): sp.set_color(C["border"])
        ax.legend(facecolor=C["card"], edgecolor=C["border"],
                  labelcolor=C["text"], fontsize=7)

    def _draw_donut(self, ax, df, y_pred=None):
        # Use predicted labels if available, else true labels
        fraud_series = (pd.Series(y_pred) if y_pred is not None
                        else df.get("is_fraud", pd.Series([0]*len(df))))
        n_total = len(fraud_series)
        n_fraud = int((fraud_series == 1).sum())
        n_legit = n_total - n_fraud

        outer_vals = np.array([n_legit, n_fraud], dtype=float)
        inner_vals = np.array([0, n_fraud], dtype=float)
        labels = ["Legitimate", "Fraud"]
        colours = [C["accent"], C["danger"]]
        fraud_c = [(0.94,0.27,0.27,0.5), (0.94,0.27,0.27,0.5)]

        ax.pie(outer_vals, radius=1.0, colors=colours, startangle=90,
               wedgeprops=dict(width=0.40, edgecolor=C["bg"], linewidth=2))
        if n_fraud > 0:
            ax.pie(inner_vals, radius=0.58, colors=[(0.1,0.4,0.9,0.3),(0.94,0.27,0.27,0.6)],
                   startangle=90, wedgeprops=dict(width=0.32, edgecolor=C["bg"], linewidth=1.5))
        ax.text(0,  0.12, f"{n_fraud:,}", ha="center", va="center",
                fontsize=14, fontweight="bold", color=C["danger"])
        ax.text(0, -0.12, "fraud", ha="center", va="center",
                fontsize=9, color=C["muted"])
        ax.text(0, -0.30, "outer=total  inner=fraud",
                ha="center", va="center", fontsize=6.5, color=C["muted"])
        pct = n_fraud/n_total*100 if n_total else 0
        patches = [mpatches.Patch(color=colours[i],
                   label=f"{labels[i]}: {outer_vals[i]:,.0f} ({outer_vals[i]/n_total*100:.2f}%)")
                   for i in range(2)]
        ax.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5,-0.22),
                  ncol=1, fontsize=8, facecolor=C["bg"], edgecolor=C["border"],
                  labelcolor=C["text"])

    def _draw_source_bar(self, ax, df):
        ax.set_facecolor(C["card"])
        if "source_tag" in df.columns:
            src = df.groupby("source_tag")["is_fraud"].agg(
                total="count", fraud=lambda x: (x==1).sum())
            x = range(len(src))
            ax.bar(x, src["total"], color=C["accent"], alpha=0.7, label="Total")
            ax.bar(x, src["fraud"], color=C["danger"],  alpha=0.9, label="Fraud")
            ax.set_xticks(list(x))
            ax.set_xticklabels(src.index.tolist(), rotation=20, fontsize=7,
                                color=C["muted"])
        else:
            ax.text(0.5,0.5,"No source info", ha="center", va="center",
                    color=C["muted"], transform=ax.transAxes)
        ax.set_title("Records by Source", color=C["text"], fontsize=9, pad=6)
        ax.tick_params(colors=C["muted"], labelsize=7)
        for sp in ax.spines.values(): sp.set_color(C["border"])
        ax.legend(facecolor=C["card"], edgecolor=C["border"],
                  labelcolor=C["text"], fontsize=7)

    def _draw_model_bar(self, ax, results):
        valid = {k:v for k,v in results.items() if "f1" in v}
        if not valid:
            ax.text(0.5,0.5,"No results", ha="center", va="center",
                    color=C["muted"], transform=ax.transAxes); return
        mets  = ["accuracy","precision","recall","f1","auc"]
        mlbls = ["Acc","Prec","Rec","F1","AUC"]
        cols  = [C["accent"],C["warn"],C["success"],C["accent2"],C["danger"]]
        x = np.arange(len(valid)); w = 0.14
        for i,(met,col,lbl) in enumerate(zip(mets,cols,mlbls)):
            ax.bar(x+i*w, [v.get(met,0) for v in valid.values()],
                   w, label=lbl, color=col, alpha=0.85)
        ax.set_xticks(x+w*2)
        ax.set_xticklabels(list(valid.keys()), fontsize=6.5,
                            color=C["muted"], rotation=8)
        ax.set_ylim(0,1.15)
        ax.set_title("Model Metrics Comparison", color=C["text"], fontsize=9, pad=6)
        ax.tick_params(colors=C["muted"], labelsize=7)
        for sp in ax.spines.values(): sp.set_color(C["border"])
        ax.legend(facecolor=C["card"],edgecolor=C["border"],
                  labelcolor=C["text"],fontsize=6,ncol=5)

    def _draw_cm(self, ax, cm, title):
        ax.set_facecolor(C["card"])
        ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_title(title, color=C["text"], fontsize=8, pad=4)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Legit","Fraud"], color=C["muted"], fontsize=7)
        ax.set_yticklabels(["Legit","Fraud"], color=C["muted"], fontsize=7)
        ax.tick_params(colors=C["muted"])
        for sp in ax.spines.values(): sp.set_color(C["border"])
        for i in range(2):
            for j in range(2):
                ax.text(j,i,f"{cm[i,j]:,}", ha="center", va="center",
                        fontsize=12, fontweight="bold",
                        color="white" if cm[i,j]>cm.max()/2 else C["text"])
        ax.set_xlabel("Predicted", color=C["muted"], fontsize=7)
        ax.set_ylabel("Actual",    color=C["muted"], fontsize=7)

    def _refresh_charts(self):
        if self._inference_preds:
            best_valid = {k:v for k,v in self.models.results.items() if "f1" in v}
            if best_valid:
                best = max(best_valid, key=lambda k: best_valid[k]["f1"])
                if best in self._inference_preds:
                    p = self._inference_preds[best]
                    self._draw_charts(y_pred=p["predictions"],
                                      y_prob=p["probabilities"])
                    return
        self._draw_charts()

    # ══════════════════════════════════════════════════════
    #  EXPORT
    # ══════════════════════════════════════════════════════

    def _export_results(self):
        if self._df_display is None:
            messagebox.showinfo("No Data","Load and score a file first."); return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV","*.csv"),("Excel","*.xlsx")])
        if not path: return
        export_df = self._df_display.copy()
        if hasattr(self, "_pred_col") and self._pred_col is not None:
            if len(self._pred_col) == len(export_df):
                export_df["fraud_predicted"] = self._pred_col
        if path.endswith(".xlsx"):
            export_df.to_excel(path, index=False)
        else:
            export_df.to_csv(path, index=False)
        messagebox.showinfo("Exported", f"Saved to:\n{path}")

    # ══════════════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════════════

    def _set_status(self, msg):
        self.status_var.set(msg)
        if self._pretrained:
            self.pretrain_badge.configure(text="✅ Models Ready", bg=C["success"])
            self.model_status_var.set("✅ Ready")
        else:
            self.pretrain_badge.configure(text="⏳ Training…", bg=C["warn"])


if __name__ == "__main__":
    app = FraudDetectorApp()
    app.mainloop()
