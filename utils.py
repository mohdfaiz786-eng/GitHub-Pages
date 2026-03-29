# utils.py - Utility Functions

import pandas as pd
import json
import base64
from io import BytesIO, StringIO
from datetime import datetime

def export_to_csv(data, filename):
    """Export data to CSV"""
    csv_buffer = StringIO()
    data.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

def export_to_excel(data, filename):
    """Export data to Excel"""
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        data.to_excel(writer, index=False, sheet_name='Predictions')
    excel_buffer.seek(0)
    return excel_buffer

def export_to_json(data, filename):
    """Export data to JSON"""
    return json.dumps(data, indent=2, default=str)

def create_download_link(content, filename, link_text):
    """Create download link for files"""
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-link">{link_text}</a>'
    return href

def format_datetime(dt):
    """Format datetime object"""
    if dt:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return "N/A"

def get_recommendations(risk_level):
    """Get recommendations based on risk level"""
    if risk_level == "High Risk":
        return [
            "🏃 Regular exercise (30 mins/day)",
            "🥗 Healthy diet with low saturated fats",
            "💊 Regular medication as prescribed",
            "🩺 Regular check-ups with cardiologist",
            "🚭 Avoid smoking and limit alcohol",
            "😴 7-8 hours of quality sleep",
            "🧘 Stress management techniques",
            "📊 Monitor blood pressure regularly"
        ]
    else:
        return [
            "🏃 Maintain regular exercise routine",
            "🥗 Balanced nutrition with fruits and vegetables",
            "😴 7-8 hours of sleep",
            "🧘 Stress management practices",
            "🩺 Annual health check-ups",
            "💧 Stay hydrated",
            "📱 Use health tracking apps",
            "🎯 Set fitness goals"
        ]