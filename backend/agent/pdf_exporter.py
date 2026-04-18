"""
PDF Exporter — generates a formatted credibility report PDF using reportlab.
"""
import io
from datetime import datetime


def export_pdf(report: dict, prediction: dict, risk_analysis: dict, raw_text: str) -> bytes:
    """
    Generate a PDF credibility report. Returns raw bytes.
    Raises ImportError if reportlab is not installed.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle("Title", parent=styles["Title"], fontSize=18, spaceAfter=6)
    h2_style = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, spaceAfter=4,
                               textColor=colors.HexColor("#1a1a2e"))
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, spaceAfter=4, leading=14)
    verdict_fake = ParagraphStyle("VerdictFake", parent=styles["Normal"], fontSize=14,
                                   textColor=colors.red, fontName="Helvetica-Bold")
    verdict_real = ParagraphStyle("VerdictReal", parent=styles["Normal"], fontSize=14,
                                   textColor=colors.green, fontName="Helvetica-Bold")
    disclaimer_style = ParagraphStyle("Disclaimer", parent=styles["Normal"], fontSize=8,
                                       textColor=colors.grey, leading=11)

    label = prediction["label"]
    confidence = prediction["confidence"]
    verdict_style = verdict_fake if label == "Fake News" else verdict_real

    story = []

    # Header
    story.append(Paragraph("🧠 News Credibility Analysis Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", body_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cccccc")))
    story.append(Spacer(1, 0.3*cm))

    # Verdict
    story.append(Paragraph("Verdict", h2_style))
    story.append(Paragraph(f"● {label}  —  {confidence}% confidence ({prediction['confidence_tier']} certainty)", verdict_style))
    story.append(Spacer(1, 0.3*cm))

    # Summary
    story.append(Paragraph("Summary", h2_style))
    story.append(Paragraph(report.get("summary", "N/A"), body_style))
    story.append(Spacer(1, 0.3*cm))

    # ML Scores table
    story.append(Paragraph("Model Scores", h2_style))
    table_data = [
        ["Metric", "Value"],
        ["Prediction", label],
        ["Confidence", f"{confidence}%"],
        ["Real Probability", f"{prediction['real_probability']}%"],
        ["Fake Probability", f"{prediction['fake_probability']}%"],
        ["Risk Score", f"{risk_analysis['risk_score']}/100"],
        ["Word Count", str(prediction['word_count'])],
    ]
    t = Table(table_data, colWidths=[6*cm, 10*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.4*cm))

    # Risk Factors
    story.append(Paragraph("Risk Factors", h2_style))
    risk_factors = report.get("risk_factors", [])
    if risk_factors:
        for rf in risk_factors:
            story.append(Paragraph(f"⚠ {rf}", body_style))
    else:
        story.append(Paragraph("No significant risk factors detected.", body_style))
    story.append(Spacer(1, 0.3*cm))

    # Credibility Indicators
    story.append(Paragraph("Credibility Indicators", h2_style))
    for ci in report.get("credibility_indicators", []):
        story.append(Paragraph(f"✓ {ci}", body_style))
    story.append(Spacer(1, 0.3*cm))

    # Cross-source verification
    story.append(Paragraph("Cross-Source Verification", h2_style))
    story.append(Paragraph(report.get("cross_source_verification", "N/A"), body_style))
    story.append(Spacer(1, 0.3*cm))

    # Confidence Assessment
    story.append(Paragraph("Confidence Assessment", h2_style))
    story.append(Paragraph(report.get("confidence_assessment", "N/A"), body_style))
    story.append(Spacer(1, 0.3*cm))

    # Sources
    sources = report.get("sources", [])
    if sources:
        story.append(Paragraph("References", h2_style))
        for s in sources:
            story.append(Paragraph(f"• {s}", body_style))
        story.append(Spacer(1, 0.3*cm))

    # Article excerpt
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph("Article Excerpt (first 300 characters)", h2_style))
    excerpt = raw_text[:300].replace("<", "&lt;").replace(">", "&gt;")
    story.append(Paragraph(f'<i>"{excerpt}..."</i>', body_style))
    story.append(Spacer(1, 0.4*cm))

    # Disclaimer
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(report.get("disclaimer", ""), disclaimer_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()
