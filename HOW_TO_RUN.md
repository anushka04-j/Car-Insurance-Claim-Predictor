# 🚀 How to Run the Application

## Method 1: Using the Batch File (Easiest)

1. **Double-click** `start_app.bat` in your project folder
2. A terminal window will open
3. Wait for the message: "You can now view your Streamlit app in your browser"
4. The app will automatically open at: **http://localhost:8501**

## Method 2: Using Command Line

1. Open **PowerShell** or **Command Prompt**
2. Navigate to your project folder:
   ```
   cd "C:\Users\anush\OneDrive\Desktop\DA project"
   ```
3. Run the command:
   ```
   python -m streamlit run app.py
   ```
4. Wait for the startup message
5. Open your browser and go to: **http://localhost:8501**

## Method 3: If Streamlit Asks for Email

If you see a prompt asking for your email:
- **Just press ENTER** (leave it blank)
- The app will continue starting

## Troubleshooting

### If the browser doesn't open automatically:
1. Open your web browser manually
2. Type in the address bar: `http://localhost:8501`
3. Press Enter

### If you see "Port 8501 is already in use":
1. Close any other Streamlit apps
2. Or use a different port:
   ```
   python -m streamlit run app.py --server.port 8502
   ```
3. Then go to: `http://localhost:8502`

### If you see "Module not found" errors:
1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### If the app shows errors:
1. Make sure you've run:
   ```
   python generate_dataset.py
   python train_model.py
   ```
2. These create the necessary data and model files

## What You Should See

When the app starts successfully, you'll see in the terminal:
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

Then your browser should open showing:
- **Header**: "🚗 On the Road Insurance"
- **Sidebar** on the left with navigation menu
- **Home page** content in the main area

## Stopping the Server

To stop the application:
- Press `Ctrl+C` in the terminal window
- Or close the terminal window

---

**Need Help?** Check the `APP_GUIDE.md` file for detailed information about each page!

