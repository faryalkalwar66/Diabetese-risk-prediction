import os
import re
from datetime import datetime
from tkinter import Tk, Toplevel, StringVar, messagebox, filedialog, ttk
import tkinter as tk

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


APP_TITLE = "Diabetes Risk Prediction System"
ADMIN_USERNAME = "faryal"
ADMIN_PASSWORD = "faryal123"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEANED_DATASET = os.path.join(BASE_DIR, "cleaned_diabetes.csv")
RAW_DATASET = os.path.join(BASE_DIR, "diabetes.csv")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET = "Outcome"

FIELD_HELP = {
    "Pregnancies": "Number of pregnancies, for example 0-20",
    "Glucose": "Plasma glucose level, for example 40-250",
    "BloodPressure": "Diastolic blood pressure, for example 30-140",
    "SkinThickness": "Skin thickness, for example 5-100",
    "Insulin": "2-hour serum insulin, for example 10-900",
    "BMI": "Body Mass Index, for example 10-70",
    "DiabetesPedigreeFunction": "Family-history diabetes score, for example 0.05-3.0",
    "Age": "Patient age, for example 1-120",
}

VALID_RANGES = {
    "Pregnancies": (0, 20),
    "Glucose": (40, 250),
    "BloodPressure": (30, 140),
    "SkinThickness": (5, 100),
    "Insulin": (10, 900),
    "BMI": (10, 70),
    "DiabetesPedigreeFunction": (0.05, 3.0),
    "Age": (1, 120),
}

COLORS = {
    "bg": "#eef7ff",
    "navy": "#0f172a",
    "blue": "#2563eb",
    "cyan": "#06b6d4",
    "green": "#16a34a",
    "orange": "#f97316",
    "red": "#dc2626",
    "purple": "#7c3aed",
    "white": "#ffffff",
    "soft": "#dbeafe",
}


class DiabetesPredictor:
    def __init__(self):
        self.dataset_path = CLEANED_DATASET if os.path.exists(CLEANED_DATASET) else RAW_DATASET
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError("diabetes.csv or cleaned_diabetes.csv was not found in the project folder.")
        self.df = self._load_and_clean_data()
        self.model = None
        self.metrics = {}
        self.threshold = 0.4
        self.train_model()

    def _load_and_clean_data(self):
        df = pd.read_csv(self.dataset_path)
        missing_cols = [col for col in FEATURES + [TARGET] if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Dataset is missing required columns: {', '.join(missing_cols)}")

        df = df[FEATURES + [TARGET]].copy()
        cols_to_fix = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        df[cols_to_fix] = df[cols_to_fix].replace(0, np.nan)
        imputer = SimpleImputer(strategy="median")
        df[cols_to_fix] = imputer.fit_transform(df[cols_to_fix])
        df[TARGET] = df[TARGET].astype(int)
        df.to_csv(CLEANED_DATASET, index=False)
        return df

    def train_model(self):
        X = self.df[FEATURES]
        y = self.df[TARGET]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Improved medical-screening model:
        # class_weight='balanced' gives more importance to diabetic cases,
        # and threshold=0.4 helps improve recall so fewer risky patients are missed.
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42,
        )
        self.model.fit(X_train, y_train)

        probabilities = self.model.predict_proba(X_test)[:, 1]
        pred = (probabilities >= self.threshold).astype(int)
        self.metrics = {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1 Score": f1_score(y_test, pred, zero_division=0),
        }

    def predict(self, patient_values):
        patient_df = pd.DataFrame([patient_values], columns=FEATURES)
        probability = float(self.model.predict_proba(patient_df)[0][1])
        prediction = 1 if probability >= self.threshold else 0
        return prediction, probability

    def append_prediction_to_dataset(self, patient_values, prediction):
        row = dict(zip(FEATURES, patient_values))
        row[TARGET] = int(prediction)
        updated = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        updated.to_csv(CLEANED_DATASET, index=False)
        self.df = updated

    def admin_summary(self):
        total = len(self.df)
        positive = int((self.df[TARGET] == 1).sum())
        negative = int((self.df[TARGET] == 0).sum())
        return {
            "Total Patients": total,
            "Predicted Diabetes": positive,
            "Predicted No Diabetes": negative,
            "High Glucose Count": int((self.df["Glucose"] >= 140).sum()),
            "Average Age": round(float(self.df["Age"].mean()), 1),
            "Average BMI": round(float(self.df["BMI"].mean()), 1),
            **{k: round(v * 100, 2) for k, v in self.metrics.items()},
        }


class MedicalGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1280x820")
        self.root.configure(bg=COLORS["bg"])
        self.root.minsize(1050, 700)

        self.predictor = DiabetesPredictor()
        self.entries = {}
        self.patient_name = StringVar()
        self.patient_contact = StringVar()
        self.last_report_data = None

        self._setup_styles()
        self.show_main_page()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", rowheight=28, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"), background=COLORS["blue"], foreground="white")
        style.map("Treeview.Heading", background=[("active", COLORS["purple"])])

    def clear(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def button(self, parent, text, command, color, width=22):
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=color,
            fg="white",
            activebackground=COLORS["navy"],
            activeforeground="white",
            font=("Segoe UI", 13, "bold"),
            relief="flat",
            bd=0,
            padx=16,
            pady=13,
            cursor="hand2",
            width=width,
        )

    def make_header(self, title, subtitle):
        header = tk.Frame(self.root, bg=COLORS["navy"], height=130)
        header.pack(fill="x")
        tk.Label(header, text="🏥  " + title, bg=COLORS["navy"], fg="white", font=("Segoe UI", 28, "bold")).pack(pady=(24, 4))
        tk.Label(header, text=subtitle, bg=COLORS["navy"], fg="#bfdbfe", font=("Segoe UI", 13)).pack()

    def show_main_page(self):
        self.clear()
        self.make_header(APP_TITLE, "Smart medical screening, patient reporting, and administration dashboard")

        main = tk.Frame(self.root, bg=COLORS["bg"])
        main.pack(expand=True, fill="both", padx=60, pady=45)

        card = tk.Frame(main, bg="white", bd=0, highlightbackground="#c7d2fe", highlightthickness=2)
        card.pack(expand=True, fill="both")

        tk.Label(card, text="Choose Your Portal", bg="white", fg=COLORS["navy"], font=("Segoe UI", 26, "bold")).pack(pady=(55, 12))
        tk.Label(card, text="Patient users can enter details and generate a PDF medical report. Administrators can view dataset and model statistics.", bg="white", fg="#475569", font=("Segoe UI", 13), wraplength=760).pack(pady=(0, 35))

        btns = tk.Frame(card, bg="white")
        btns.pack(pady=10)
        self.button(btns, "👤 Patient View", self.show_patient_form, COLORS["green"], 24).grid(row=0, column=0, padx=25, pady=10)
        self.button(btns, "🔐 Administration Login", self.show_admin_login, COLORS["purple"], 24).grid(row=0, column=1, padx=25, pady=10)


    def show_patient_form(self):
        self.clear()
        self.make_header("Patient Prediction Form", "Enter correct medical values to generate a diabetes risk report")

        outer = tk.Frame(self.root, bg=COLORS["bg"])
        outer.pack(fill="both", expand=True, padx=30, pady=20)

        # Pack the guide panel first and keep a fixed width so it never gets pushed/cut off.
        side = tk.Frame(outer, bg=COLORS["soft"], width=390)
        side.pack(side="right", fill="y", padx=(18, 0))
        side.pack_propagate(False)

        form = tk.Frame(outer, bg="white", highlightbackground="#93c5fd", highlightthickness=2)
        form.pack(side="left", fill="both", expand=True)

        tk.Label(form, text="Patient Information", bg="white", fg=COLORS["blue"], font=("Segoe UI", 20, "bold")).grid(row=0, column=0, columnspan=4, pady=(22, 12))

        labels = [("Patient Name", self.patient_name), ("Contact Number", self.patient_contact)]
        for i, (label, var) in enumerate(labels):
            tk.Label(form, text=label, bg="white", fg=COLORS["navy"], font=("Segoe UI", 11, "bold")).grid(row=1, column=i * 2, padx=14, pady=8, sticky="e")
            tk.Entry(form, textvariable=var, font=("Segoe UI", 11), bd=1, relief="solid", width=22).grid(row=1, column=i * 2 + 1, padx=14, pady=8, sticky="w")

        self.entries = {}
        for idx, feature in enumerate(FEATURES):
            row = 2 + idx // 2
            col = (idx % 2) * 2
            tk.Label(form, text=feature, bg="white", fg=COLORS["navy"], font=("Segoe UI", 11, "bold")).grid(row=row, column=col, padx=14, pady=12, sticky="e")
            entry = tk.Entry(form, font=("Segoe UI", 11), bd=1, relief="solid", width=22)
            entry.grid(row=row, column=col + 1, padx=14, pady=12, sticky="w")
            self.entries[feature] = entry

        actions = tk.Frame(form, bg="white")
        actions.grid(row=7, column=0, columnspan=4, pady=28)
        self.button(actions, "🔎 Predict & Save", self.predict_patient, COLORS["green"], 16).grid(row=0, column=0, padx=8)
        self.button(actions, "🧹 Clear Form", self.clear_patient_form, COLORS["cyan"], 14).grid(row=0, column=1, padx=8)
        self.button(actions, "📄 Download PDF", self.download_last_report, COLORS["orange"], 16).grid(row=0, column=2, padx=8)
        self.button(actions, "🚪 Logout", self.show_main_page, COLORS["red"], 12).grid(row=0, column=3, padx=8)

        tk.Label(side, text="Input Guide", bg=COLORS["soft"], fg=COLORS["navy"], font=("Segoe UI", 18, "bold")).pack(pady=(24, 10))

        guide_box = tk.Text(
            side,
            bg=COLORS["soft"],
            fg="#334155",
            font=("Segoe UI", 10),
            wrap="word",
            relief="flat",
            bd=0,
            padx=18,
            pady=8,
            height=23,
            width=42,
            cursor="arrow",
        )
        guide_scroll = ttk.Scrollbar(side, orient="vertical", command=guide_box.yview)
        guide_box.configure(yscrollcommand=guide_scroll.set)
        guide_box.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=(0, 12))
        guide_scroll.pack(side="right", fill="y", padx=(0, 8), pady=(0, 12))

        guide_lines = []
        for feature, help_text in FIELD_HELP.items():
            guide_lines.append(f"• {feature}: {help_text}")
        guide_box.insert("1.0", "\n\n".join(guide_lines))
        guide_box.configure(state="disabled")

    def validate_form(self):
        name = self.patient_name.get().strip()
        contact = self.patient_contact.get().strip()
        if not name or len(name) < 2:
            messagebox.showerror("Correction Needed", "Please enter a valid patient name.")
            return None
        if not re.fullmatch(r"[0-9+\- ]{7,18}", contact):
            messagebox.showerror("Correction Needed", "Please enter a valid contact number, for example 03001234567.")
            return None

        values = []
        for feature, entry in self.entries.items():
            raw = entry.get().strip()
            try:
                value = float(raw)
            except ValueError:
                messagebox.showerror("Correction Needed", f"{feature} must be a numeric value.\n{FIELD_HELP[feature]}")
                entry.focus_set()
                return None
            low, high = VALID_RANGES[feature]
            if not (low <= value <= high):
                messagebox.showerror("Correction Needed", f"{feature} should be between {low} and {high}.\n{FIELD_HELP[feature]}")
                entry.focus_set()
                return None
            if feature in ["Pregnancies", "Age"] and value != int(value):
                messagebox.showerror("Correction Needed", f"{feature} should be a whole number.")
                entry.focus_set()
                return None
            values.append(int(value) if feature in ["Pregnancies", "Age"] else value)
        return name, contact, values

    def clear_patient_form(self):
        self.patient_name.set("")
        self.patient_contact.set("")
        for entry in self.entries.values():
            entry.delete(0, tk.END)
        self.last_report_data = None
        messagebox.showinfo("Form Cleared", "All patient form fields have been cleared.")

    def predict_patient(self):
        validated = self.validate_form()
        if not validated:
            return
        name, contact, values = validated
        prediction, probability = self.predictor.predict(values)
        self.predictor.append_prediction_to_dataset(values, prediction)

        risk_percent = round(probability * 100, 2)
        if risk_percent >= 70:
            risk_level = "High Risk"
            alert_color = COLORS["red"]
            consult = "Consult a doctor as soon as possible."
        elif risk_percent >= 40:
            risk_level = "Medium Risk"
            alert_color = COLORS["orange"]
            consult = "Schedule a medical checkup and discuss the report with a doctor."
        else:
            risk_level = "Low Risk"
            alert_color = COLORS["green"]
            consult = "Maintain healthy habits and continue routine checkups."

        result_text = "Diabetes Risk Detected" if prediction == 1 else "No Diabetes Risk Detected"
        explanation = (
            "The model compares the patient's medical attributes with previous patient records. "
            "Higher glucose, BMI, age, insulin, and family-history score can increase diabetes risk. "
            "This is a screening result, not a final medical diagnosis."
        )
        lifestyle = [
            "Reduce salt and added sugar intake.",
            "Exercise at least 30 minutes daily.",
            "Schedule a medical checkup.",
            "Maintain a balanced diet with vegetables, whole grains, and lean protein.",
            "Monitor glucose level regularly if symptoms or family history exist.",
        ]

        self.last_report_data = {
            "name": name,
            "contact": contact,
            "values": dict(zip(FEATURES, values)),
            "prediction": result_text,
            "risk_percent": risk_percent,
            "risk_level": risk_level,
            "consult": consult,
            "explanation": explanation,
            "lifestyle": lifestyle,
            "timestamp": datetime.now().strftime("%d-%m-%Y %I:%M %p"),
        }

        messagebox.showinfo(
            "Prediction Complete",
            f"Result: {result_text}\nRisk Probability: {risk_percent}%\nAlert: {risk_level}\nRecommendation: {consult}",
        )
        self.show_result_page(alert_color)

    def show_result_page(self, alert_color):
        self.clear()
        data = self.last_report_data
        self.make_header("Patient Medical Report", "Prediction completed successfully")

        card = tk.Frame(self.root, bg="white", highlightbackground=alert_color, highlightthickness=4)
        card.pack(expand=True, fill="both", padx=70, pady=35)

        tk.Label(card, text=data["risk_level"], bg="white", fg=alert_color, font=("Segoe UI", 30, "bold")).pack(pady=(30, 5))
        tk.Label(card, text=data["prediction"], bg="white", fg=COLORS["navy"], font=("Segoe UI", 20, "bold")).pack(pady=5)
        tk.Label(card, text=f"Diabetes Probability: {data['risk_percent']}%", bg="white", fg="#334155", font=("Segoe UI", 16)).pack(pady=5)
        tk.Label(card, text=data["consult"], bg="white", fg="#334155", font=("Segoe UI", 13), wraplength=800).pack(pady=10)

        btns = tk.Frame(card, bg="white")
        btns.pack(pady=28)
        self.button(btns, "📄 Download PDF Report", self.download_last_report, COLORS["orange"], 22).grid(row=0, column=0, padx=12)
        self.button(btns, "➕ New Patient", self.show_patient_form, COLORS["blue"], 16).grid(row=0, column=1, padx=12)
        self.button(btns, "🚪 Logout", self.show_main_page, COLORS["red"], 14).grid(row=0, column=2, padx=12)

    def download_last_report(self):
        if not self.last_report_data:
            messagebox.showwarning("No Report", "Please predict a patient result first.")
            return
        default_name = f"medical_report_{self.last_report_data['name'].replace(' ', '_')}.pdf"
        save_path = filedialog.asksaveasfilename(
            title="Save Medical Report",
            defaultextension=".pdf",
            initialdir=REPORT_DIR,
            initialfile=default_name,
            filetypes=[("PDF Files", "*.pdf")],
        )
        if not save_path:
            return
        self.create_pdf_report(save_path)
        messagebox.showinfo("Report Saved", f"PDF medical report saved successfully:\n{save_path}")

    def create_pdf_report(self, path):
        data = self.last_report_data
        doc = SimpleDocTemplate(path, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("<b>Diabetes Risk Medical Report</b>", styles["Title"]))
        story.append(Paragraph("Smart Medical Screening System", styles["Heading2"]))
        story.append(Spacer(1, 8))
        story.append(Paragraph(f"<b>Generated:</b> {data['timestamp']}", styles["Normal"]))
        story.append(Paragraph("<b>Report Type:</b> Machine Learning Based Diabetes Risk Screening", styles["Normal"]))
        story.append(Spacer(1, 12))

        patient_table = [
            ["Patient Name", data["name"]],
            ["Contact Number", data["contact"]],
            ["Prediction", data["prediction"]],
            ["Risk Level", data["risk_level"]],
            ["Diabetes Probability", f"{data['risk_percent']}%"],
            ["Doctor Recommendation", data["consult"]],
        ]
        story.append(self._styled_table(patient_table, [170, 330]))
        story.append(Spacer(1, 16))

        story.append(Paragraph("Patient Input Data", styles["Heading2"]))
        input_table = [["Attribute", "Value"]] + [[k, str(v)] for k, v in data["values"].items()]
        story.append(self._styled_table(input_table, [230, 270], header=True))
        story.append(Spacer(1, 16))

        story.append(Paragraph("Simple Explanation", styles["Heading2"]))
        story.append(Paragraph(data["explanation"], styles["BodyText"]))
        story.append(Spacer(1, 12))

        story.append(Paragraph("Lifestyle Recommendations", styles["Heading2"]))
        for item in data["lifestyle"]:
            story.append(Paragraph(f"• {item}", styles["BodyText"]))

        story.append(Spacer(1, 16))
        story.append(Paragraph("Note: This system is for educational screening support only. A qualified doctor should confirm any diagnosis.", styles["Italic"]))
        doc.build(story)

    def _styled_table(self, rows, widths, header=False):
        table = Table(rows, colWidths=widths)
        style = [
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#2563eb")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bfdbfe")),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#dbeafe")),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#0f172a")),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]
        if header:
            style += [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563eb")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ]
        table.setStyle(TableStyle(style))
        return table

    def show_admin_login(self):
        login = Toplevel(self.root)
        login.title("Admin Login")
        login.geometry("390x310")
        login.configure(bg="white")
        login.resizable(False, False)
        login.grab_set()

        username = StringVar()
        password = StringVar()

        tk.Label(login, text="🔐 Administrator Login", bg="white", fg=COLORS["purple"], font=("Segoe UI", 20, "bold")).pack(pady=(28, 16))
        tk.Label(login, text="Username", bg="white", fg=COLORS["navy"], font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=52)
        tk.Entry(login, textvariable=username, font=("Segoe UI", 11), bd=1, relief="solid").pack(fill="x", padx=52, pady=(4, 12))
        tk.Label(login, text="Password", bg="white", fg=COLORS["navy"], font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=52)
        tk.Entry(login, textvariable=password, show="*", font=("Segoe UI", 11), bd=1, relief="solid").pack(fill="x", padx=52, pady=(4, 16))

        def check_login():
            if username.get().strip() == ADMIN_USERNAME and password.get().strip() == ADMIN_PASSWORD:
                login.destroy()
                self.show_admin_dashboard()
            else:
                messagebox.showerror("Access Denied", "Wrong username or password.")

        self.button(login, "Login", check_login, COLORS["purple"], 16).pack(pady=5)

    def show_admin_dashboard(self):
        self.clear()
        self.make_header("Administration Dashboard", "Dataset statistics, patient counts, charts, model performance, and recent records")

        # Top-right logout button for the administration portal.
        self.button(self.root, "🚪 Logout", self.show_main_page, COLORS["red"], 10).place(relx=0.97, y=28, anchor="ne")

        # Scrollable dashboard area: on small laptop screens the recent records were going below the window.
        # Now the full admin dashboard can be scrolled vertically.
        wrapper = tk.Frame(self.root, bg=COLORS["bg"])
        wrapper.pack(fill="both", expand=True)

        canvas = tk.Canvas(wrapper, bg=COLORS["bg"], highlightthickness=0)
        page_scroll = ttk.Scrollbar(wrapper, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=page_scroll.set)
        page_scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        body = tk.Frame(canvas, bg=COLORS["bg"])
        body_window = canvas.create_window((0, 0), window=body, anchor="nw")

        def _update_scroll_region(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfigure(body_window, width=canvas.winfo_width())

        body.bind("<Configure>", _update_scroll_region)
        canvas.bind("<Configure>", _update_scroll_region)

        def _mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _mousewheel)

        inner = tk.Frame(body, bg=COLORS["bg"])
        inner.pack(fill="both", expand=True, padx=22, pady=12)

        summary = self.predictor.admin_summary()

        stats_frame = tk.Frame(inner, bg=COLORS["bg"])
        stats_frame.pack(fill="x")
        palette = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["purple"], COLORS["cyan"], COLORS["red"], COLORS["navy"], "#0891b2", "#9333ea", "#ca8a04"]
        for i, (label, value) in enumerate(summary.items()):
            card = tk.Frame(stats_frame, bg=palette[i % len(palette)], width=150, height=72)
            card.grid(row=i // 5, column=i % 5, padx=7, pady=6, sticky="nsew")
            card.grid_propagate(False)
            tk.Label(card, text=str(value), bg=palette[i % len(palette)], fg="white", font=("Segoe UI", 16, "bold")).pack(pady=(8, 0))
            tk.Label(card, text=label, bg=palette[i % len(palette)], fg="white", font=("Segoe UI", 8, "bold"), wraplength=135).pack()

        charts_frame = tk.Frame(inner, bg="white", highlightbackground="#bfdbfe", highlightthickness=2)
        charts_frame.pack(fill="x", pady=(8, 10))
        tk.Label(charts_frame, text="Visual Analytics", bg="white", fg=COLORS["navy"], font=("Segoe UI", 14, "bold")).pack(pady=(6, 0))

        canvases = tk.Frame(charts_frame, bg="white")
        canvases.pack(fill="x", padx=8, pady=5)
        pie = tk.Canvas(canvases, width=330, height=155, bg="white", highlightthickness=0)
        pie.pack(side="left", padx=12)
        bars = tk.Canvas(canvases, width=520, height=155, bg="white", highlightthickness=0)
        bars.pack(side="left", padx=12, fill="x", expand=True)
        self.draw_pie_chart(pie, int(summary["Predicted Diabetes"]), int(summary["Predicted No Diabetes"]))
        self.draw_bar_chart(bars, {
            "High Glucose": int(summary["High Glucose Count"]),
            "Avg BMI": float(summary["Average BMI"]),
            "Avg Age": float(summary["Average Age"]),
            "Diabetic": int(summary["Predicted Diabetes"]),
        })

        table_frame = tk.Frame(inner, bg="white", highlightbackground="#94a3b8", highlightthickness=2)
        table_frame.pack(fill="both", expand=True, pady=(0, 8))
        tk.Label(table_frame, text="Recent Patient Dataset Records", bg="white", fg=COLORS["navy"], font=("Segoe UI", 15, "bold")).pack(pady=8)
        tk.Label(table_frame, text="Scroll down / use mouse wheel to see all records", bg="white", fg="#64748b", font=("Segoe UI", 9)).pack(pady=(0, 4))

        tree_container = tk.Frame(table_frame, bg="white")
        tree_container.pack(fill="both", expand=True, padx=10, pady=6)
        tree = ttk.Treeview(tree_container, columns=FEATURES + [TARGET], show="headings", height=12)
        vsb = ttk.Scrollbar(tree_container, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(tree_container, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        for col in FEATURES + [TARGET]:
            tree.heading(col, text=col)
            tree.column(col, width=130, anchor="center", stretch=False)
        for _, row in self.predictor.df.tail(40).iterrows():
            tree.insert("", "end", values=[row[col] for col in FEATURES + [TARGET]])
        tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")

        footer = tk.Frame(inner, bg=COLORS["bg"])
        footer.pack(fill="x", pady=(0, 10))
        self.button(footer, "🔄 Refresh Dashboard", self.show_admin_dashboard, COLORS["blue"], 20).pack(side="left", padx=8, pady=5)

    def draw_pie_chart(self, canvas, diabetic, non_diabetic):
        total = max(diabetic + non_diabetic, 1)
        diabetic_extent = 360 * diabetic / total
        canvas.create_text(165, 18, text="Diabetic vs Non-Diabetic", fill=COLORS["navy"], font=("Segoe UI", 12, "bold"))
        canvas.create_arc(55, 38, 165, 148, start=90, extent=diabetic_extent, fill=COLORS["red"], outline="white")
        canvas.create_arc(55, 38, 165, 148, start=90 + diabetic_extent, extent=360 - diabetic_extent, fill=COLORS["green"], outline="white")
        canvas.create_rectangle(190, 62, 206, 78, fill=COLORS["red"], outline="")
        canvas.create_text(214, 70, text=f"Diabetic: {diabetic}", anchor="w", fill="#334155", font=("Segoe UI", 10, "bold"))
        canvas.create_rectangle(190, 92, 206, 108, fill=COLORS["green"], outline="")
        canvas.create_text(214, 100, text=f"Non-Diabetic: {non_diabetic}", anchor="w", fill="#334155", font=("Segoe UI", 10, "bold"))

    def draw_bar_chart(self, canvas, data):
        canvas.create_text(260, 18, text="Quick Health Indicators", fill=COLORS["navy"], font=("Segoe UI", 12, "bold"))
        max_value = max(data.values()) or 1
        x = 55
        colors_list = [COLORS["purple"], COLORS["orange"], COLORS["cyan"], COLORS["blue"]]
        for i, (label, value) in enumerate(data.items()):
            bar_h = int((value / max_value) * 95)
            x0 = x + i * 110
            canvas.create_rectangle(x0, 135 - bar_h, x0 + 55, 135, fill=colors_list[i], outline="")
            canvas.create_text(x0 + 27, 126 - bar_h, text=str(round(value, 1)), fill=COLORS["navy"], font=("Segoe UI", 9, "bold"))
            canvas.create_text(x0 + 27, 150, text=label, fill="#334155", font=("Segoe UI", 9, "bold"))


def main():
    root = Tk()
    try:
        MedicalGUI(root)
    except Exception as exc:
        messagebox.showerror("Application Error", str(exc))
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
