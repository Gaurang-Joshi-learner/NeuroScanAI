from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)
from reportlab.lib.styles import getSampleStyleSheet


def generate_seizure_report(filepath, analysis):
    doc = SimpleDocTemplate(filepath)
    styles = getSampleStyleSheet()

    result = analysis.result or {}

    risk_level = result.get("risk_level", "Unknown")

    overall_risk = round(
        result.get("overall_risk_score", 0) * 100,
        2
    )

    seizure_epochs = result.get("n_seizure_epochs", 0)

    total_epochs = result.get("n_epochs", 0)

    duration = result.get("recording_duration_sec", 0)

    channels = result.get("n_channels", 0)

    elements = []

    elements.append(
        Paragraph(
            "NeuroScanAI Seizure Analysis Report",
            styles["Title"]
        )
    )

    elements.append(Spacer(1, 20))

    elements.append(
        Paragraph(
            f"Filename: {analysis.filename}",
            styles["BodyText"]
        )
    )

    elements.append(
        Paragraph(
            f"Analysis Type: {analysis.analysis_type.value}",
            styles["BodyText"]
        )
    )

    elements.append(
        Paragraph(
            f"Risk Level: {risk_level}",
            styles["BodyText"]
        )
    )

    elements.append(
        Paragraph(
            f"Overall Risk Score: {overall_risk}%",
            styles["BodyText"]
        )
    )

    elements.append(
        Paragraph(
            f"Detected Seizure Epochs: {seizure_epochs}",
            styles["BodyText"]
        )
    )

    elements.append(
        Paragraph(
            f"Total Epochs: {total_epochs}",
            styles["BodyText"]
        )
    )

    elements.append(
        Paragraph(
            f"Recording Duration: {duration} seconds",
            styles["BodyText"]
        )
    )

    elements.append(
        Paragraph(
            f"EEG Channels: {channels}",
            styles["BodyText"]
        )
    )

    doc.build(elements)