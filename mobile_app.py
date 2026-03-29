# mobile_app.py - Simple Version
import flet as ft
import requests

def main(page: ft.Page):
    page.title = "CardioAI Pro"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = "#0f172a"
    page.padding = 20
    
    API_URL = "https://heart-disease-predictor-1-c03z.onrender.com/predict"
    
    # Inputs
    age = ft.TextField(label="Age", value="55", bgcolor="#1e293b")
    chol = ft.TextField(label="Cholesterol", value="220", bgcolor="#1e293b")
    bp = ft.TextField(label="Blood Pressure", value="130", bgcolor="#1e293b")
    result = ft.Text("", size=20)
    
    def predict_click(e):
        try:
            data = {
                "age": int(age.value),
                "chol": int(chol.value),
                "trestbps": int(bp.value)
            }
            resp = requests.post(API_URL, json=data)
            res = resp.json()
            
            if res.get("prediction") == 1:
                result.value = f"⚠️ High Risk: {res.get('risk_percentage')}%"
                result.color = "red"
            else:
                result.value = f"✅ Low Risk: {100 - res.get('risk_percentage')}%"
                result.color = "green"
        except:
            result.value = "❌ Error connecting to API"
            result.color = "red"
        
        page.update()
    
    btn = ft.ElevatedButton("Analyze", on_click=predict_click, bgcolor="#3b82f6")
    
    page.add(
        ft.Text("❤️ CardioAI Pro", size=28, color="#60a5fa"),
        age, chol, bp, btn, result
    )

ft.app(target=main)