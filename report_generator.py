# report_generator.py - Complete Working Report Generator

import pandas as pd
from datetime import datetime
import base64

class ReportGenerator:
    def __init__(self):
        pass
    
    def generate_html_report(self, results_df, dataset_info, feature_importance, best_model):
        """Generate HTML report with proper error handling"""
        
        # Handle results_df safely
        if results_df is not None and isinstance(results_df, pd.DataFrame) and len(results_df) > 0:
            # Remove problematic columns
            display_cols = []
            for col in results_df.columns:
                if col not in ['Model Object', 'Accuracy_Score', 'Unnamed: 0']:
                    display_cols.append(col)
            results_html = results_df[display_cols].to_html(index=False, escape=False)
        else:
            results_html = "<p style='color: black;'>No results available. Please train models first.</p>"
        
        # Handle feature importance safely
        if feature_importance is not None and isinstance(feature_importance, pd.DataFrame) and len(feature_importance) > 0:
            if 'Feature' in feature_importance.columns:
                imp_display = feature_importance[['Feature']].head(10).copy()
                if 'Importance' in feature_importance.columns:
                    imp_display['Importance'] = feature_importance['Importance'].head(10)
                elif 'MI_Score' in feature_importance.columns:
                    imp_display['Importance'] = feature_importance['MI_Score'].head(10)
                else:
                    imp_display['Importance'] = 0
                feature_html = imp_display.to_html(index=False)
            else:
                feature_html = "<p>Feature importance data not available</p>"
        else:
            feature_html = "<p>No feature importance data available. Run feature selection first.</p>"
        
        # Safely get dataset values
        samples = dataset_info.get('samples', 0) if isinstance(dataset_info, dict) else 0
        features = dataset_info.get('features', 0) if isinstance(dataset_info, dict) else 0
        target = dataset_info.get('target', 'N/A') if isinstance(dataset_info, dict) else 'N/A'
        positive = dataset_info.get('positive', 0) if isinstance(dataset_info, dict) else 0
        negative = dataset_info.get('negative', 0) if isinstance(dataset_info, dict) else 0
        
        # Best model
        best_model_name = best_model if best_model else "Not Available"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CardioAI Pro - Heart Disease Report</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                
                body {{
                    font-family: 'Segoe UI', 'Poppins', system-ui, -apple-system, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 40px 20px;
                }}
                
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 24px;
                    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                    overflow: hidden;
                    animation: fadeIn 0.5s ease-out;
                }}
                
                @keyframes fadeIn {{
                    from {{ opacity: 0; transform: translateY(20px); }}
                    to {{ opacity: 1; transform: translateY(0); }}
                }}
                
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 40px;
                    text-align: center;
                }}
                
                .header h1 {{
                    font-size: 2.5rem;
                    margin-bottom: 10px;
                }}
                
                .header p {{
                    opacity: 0.9;
                    font-size: 1rem;
                }}
                
                .content {{
                    padding: 40px;
                }}
                
                .section {{
                    margin-bottom: 40px;
                }}
                
                .section-title {{
                    color: #667eea;
                    font-size: 1.5rem;
                    margin-bottom: 20px;
                    padding-bottom: 10px;
                    border-bottom: 3px solid #667eea;
                }}
                
                .info-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                
                .info-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 16px;
                    text-align: center;
                    transition: transform 0.3s;
                }}
                
                .info-card:hover {{
                    transform: translateY(-5px);
                }}
                
                .info-card h3 {{
                    font-size: 2rem;
                    margin-bottom: 8px;
                }}
                
                .info-card p {{
                    font-size: 0.85rem;
                    opacity: 0.9;
                }}
                
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                    background: white;
                    border-radius: 12px;
                    overflow: hidden;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                
                th {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 14px;
                    text-align: left;
                    font-weight: 600;
                }}
                
                td {{
                    padding: 12px 14px;
                    border-bottom: 1px solid #e5e7eb;
                    color: #374151;
                }}
                
                tr:hover {{
                    background: #f9fafb;
                }}
                
                .best-model-card {{
                    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                    color: white;
                    padding: 25px;
                    border-radius: 16px;
                    text-align: center;
                    margin: 20px 0;
                }}
                
                .best-model-card h3 {{
                    font-size: 1.5rem;
                    margin-bottom: 10px;
                }}
                
                .recommendations {{
                    background: #f3f4f6;
                    padding: 25px;
                    border-radius: 16px;
                    margin: 20px 0;
                }}
                
                .recommendations h4 {{
                    color: #374151;
                    margin-bottom: 15px;
                    font-size: 1.2rem;
                }}
                
                .recommendations ul {{
                    list-style: none;
                    padding-left: 0;
                }}
                
                .recommendations li {{
                    padding: 8px 0;
                    padding-left: 28px;
                    position: relative;
                    color: #4b5563;
                }}
                
                .recommendations li:before {{
                    content: "✓";
                    color: #10b981;
                    position: absolute;
                    left: 0;
                    font-weight: bold;
                    font-size: 1.1rem;
                }}
                
                .footer {{
                    background: #f9fafb;
                    padding: 20px 40px;
                    text-align: center;
                    color: #6b7280;
                    font-size: 0.8rem;
                    border-top: 1px solid #e5e7eb;
                }}
                
                @media (max-width: 768px) {{
                    .header h1 {{ font-size: 1.8rem; }}
                    .content {{ padding: 20px; }}
                    .info-grid {{ grid-template-columns: 1fr; }}
                    table {{ font-size: 0.8rem; }}
                    th, td {{ padding: 8px; }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>❤️ CardioAI Pro</h1>
                    <p>Advanced Heart Disease Prediction Report</p>
                    <p style="font-size: 0.85rem; margin-top: 10px;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="content">
                    <div class="section">
                        <h2 class="section-title">📊 Dataset Overview</h2>
                        <div class="info-grid">
                            <div class="info-card">
                                <h3>{samples}</h3>
                                <p>Total Samples</p>
                            </div>
                            <div class="info-card">
                                <h3>{features}</h3>
                                <p>Total Features</p>
                            </div>
                            <div class="info-card">
                                <h3>{positive}</h3>
                                <p>Positive Cases</p>
                            </div>
                            <div class="info-card">
                                <h3>{negative}</h3>
                                <p>Negative Cases</p>
                            </div>
                        </div>
                        <p style="color: #6b7280; margin-top: 10px;"><strong>Target Column:</strong> {target}</p>
                    </div>
                    
                    <div class="section">
                        <h2 class="section-title">🤖 Model Performance Comparison</h2>
                        {results_html}
                    </div>
                    
                    <div class="best-model-card">
                        <h3>🏆 Best Performing Model</h3>
                        <p style="font-size: 1.2rem; margin-top: 8px;">{best_model_name}</p>
                    </div>
                    
                    <div class="section">
                        <h2 class="section-title">📈 Feature Importance Analysis</h2>
                        {feature_html}
                    </div>
                    
                    <div class="section">
                        <h2 class="section-title">💊 Medical Recommendations</h2>
                        <div class="recommendations">
                            <h4>Preventive Measures & Healthy Lifestyle</h4>
                            <ul>
                                <li><strong>Regular Exercise:</strong> At least 30 minutes of moderate activity daily</li>
                                <li><strong>Heart-Healthy Diet:</strong> Low saturated fats, high fiber, plenty of fruits and vegetables</li>
                                <li><strong>Regular Check-ups:</strong> Annual health screening and cardiac check-ups</li>
                                <li><strong>Stress Management:</strong> Practice meditation, yoga, or deep breathing exercises</li>
                                <li><strong>Quality Sleep:</strong> 7-8 hours of sleep per night</li>
                                <li><strong>No Smoking:</strong> Avoid tobacco and limit alcohol consumption</li>
                                <li><strong>Monitor BP:</strong> Regular blood pressure monitoring</li>
                                <li><strong>Medication Adherence:</strong> Take prescribed medicines regularly</li>
                            </ul>
                        </div>
                    </div>
                </div>
                
                <div class="footer">
                    <p>⚠️ This report is generated by CardioAI Pro - AI-powered heart disease prediction system</p>
                    <p>📌 For medical advice, please consult a qualified healthcare professional</p>
                    <p>🔬 Report ID: CAR-{datetime.now().strftime('%Y%m%d%H%M%S')}</p>
                </div>
            </div>
        </body>
        </html>
        """
        return html
    
    def generate_csv_report(self, dataset_info, feature_importance, results_df):
        """Generate CSV report with proper handling"""
        import io
        
        output = io.StringIO()
        
        # Header
        output.write(f"CardioAI Pro Report\n")
        output.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Dataset Information
        output.write("DATASET INFORMATION\n")
        if isinstance(dataset_info, dict):
            output.write(f"Total Samples,{dataset_info.get('samples', 0)}\n")
            output.write(f"Total Features,{dataset_info.get('features', 0)}\n")
            output.write(f"Target Column,{dataset_info.get('target', 'N/A')}\n")
            output.write(f"Positive Cases,{dataset_info.get('positive', 0)}\n")
            output.write(f"Negative Cases,{dataset_info.get('negative', 0)}\n")
        output.write("\n")
        
        # Model Performance
        if results_df is not None and isinstance(results_df, pd.DataFrame) and len(results_df) > 0:
            output.write("MODEL PERFORMANCE\n")
            # Remove problematic columns
            cols_to_drop = ['Model Object', 'Accuracy_Score']
            results_clean = results_df.drop(columns=[c for c in cols_to_drop if c in results_df.columns], errors='ignore')
            results_clean.to_csv(output, index=False)
            output.write("\n")
        
        # Feature Importance
        if feature_importance is not None and isinstance(feature_importance, pd.DataFrame) and len(feature_importance) > 0:
            output.write("FEATURE IMPORTANCE\n")
            feature_importance.to_csv(output, index=False)
        
        return output.getvalue()