

# pdf_generator.py - PDF Report Generation

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
from io import BytesIO
from datetime import datetime

class PDFReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        """Setup custom styles for PDF"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#667eea'),
            alignment=TA_CENTER,
            spaceAfter=30
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#764ba2'),
            spaceAfter=12
        ))
        
        self.styles.add(ParagraphStyle(
            name='RiskHigh',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=colors.HexColor('#dc2626'),
            alignment=TA_CENTER,
            spaceAfter=10
        ))
        
        self.styles.add(ParagraphStyle(
            name='RiskLow',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=colors.HexColor('#059669'),
            alignment=TA_CENTER,
            spaceAfter=10
        ))
    
    def generate_report(self, patient_info, prediction_data, risk_percentage, recommendations):
        """Generate professional PDF report"""
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, 
                               rightMargin=72, leftMargin=72, 
                               topMargin=72, bottomMargin=72)
        
        story = []
        
        # Title
        title = Paragraph("CardioAI - Heart Disease Assessment Report", 
                         self.styles['CustomTitle'])
        story.append(title)
        
        # Date
        date_para = Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                             self.styles['Normal'])
        story.append(date_para)
        story.append(Spacer(1, 20))
        
        # Patient Information
        story.append(Paragraph("Patient Information", self.styles['CustomHeading']))
        
        patient_table_data = []
        for key, value in patient_info.items():
            if key != 'patient_data':
                patient_table_data.append([key.replace('_', ' ').title(), str(value)])
        
        patient_table = Table(patient_table_data, colWidths=[2*inch, 3.5*inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(patient_table)
        story.append(Spacer(1, 20))
        
        # Prediction Results
        story.append(Paragraph("Prediction Results", self.styles['CustomHeading']))
        
        risk_level = "HIGH RISK" if risk_percentage > 50 else "LOW RISK"
        risk_style = 'RiskHigh' if risk_percentage > 50 else 'RiskLow'
        
        risk_para = Paragraph(f"Risk Assessment: {risk_level}", 
                             self.styles[risk_style])
        story.append(risk_para)
        
        results_data = [
            ["Risk Probability", f"{risk_percentage:.2f}%"],
            ["Confidence Level", "High"],
            ["Prediction Time", datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        ]
        
        results_table = Table(results_data, colWidths=[2*inch, 3.5*inch])
        results_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(results_table)
        story.append(Spacer(1, 20))
        
        # Recommendations
        story.append(Paragraph("Medical Recommendations", self.styles['CustomHeading']))
        for rec in recommendations:
            rec_para = Paragraph(f"• {rec}", self.styles['Normal'])
            story.append(rec_para)
        
        story.append(Spacer(1, 20))
        
        # Footer
        footer_text = "This report is generated by CardioAI - Advanced Heart Disease Prediction System. "
        footer_text += "Please consult a healthcare professional for medical advice."
        footer = Paragraph(footer_text, self.styles['Normal'])
        story.append(footer)
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer