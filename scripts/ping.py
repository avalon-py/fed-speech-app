import sys
import urllib.request
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# Replace with your actual Streamlit URL
URL = "https://fed-speech-app.streamlit.app/"

def is_app_sleeping(url):
    """Performs a fast HTTP request to check if the app is asleep."""
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read().decode('utf-8', errors='ignore')
            
            # Streamlit's fallback sleep page contains these markers or relies heavily on JS
            if "enable javascript" in html.lower() or "zzzz" in html.lower():
                print("⚠️ HTML hint indicates the app might be sleeping or requiring JS.")
                return True
            
            print(f"✅ Simple ping successful! Status: {response.getcode()}. Timer reset.")
            return False
    except Exception as e:
        print(f"⚠️ Simple ping failed or timed out ({e}). Resorting to browser check...")
        return True

def wake_up_with_selenium(url):
    """Launches a headless browser to forcefully click the wake button if needed."""
    print("🚀 Launching Headless Chrome via Selenium...")
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=chrome_options)
    try:
        driver.get(url)
        time.sleep(6)  # Give the JS page a moment to render the sleep dialog

        # Find any buttons on the screen
        buttons = driver.find_elements(By.TAG_NAME, "button")
        wake_button = None
        
        for btn in buttons:
            btn_text = btn.text.lower()
            if "get this app back up" in btn_text or "wake" in btn_text:
                wake_button = btn
                break

        if wake_button:
            print("😴 App confirmed ASLEEP! Clicking 'Yes, get this app back up!'...")
            wake_button.click()
            time.sleep(15)  # Wait for backend container spin-up
            print("🎉 Wake up signal sent successfully.")
        else:
            print("😎 App is verified AWAKE in browser view. No action required.")

    except Exception as e:
        print(f"❌ Selenium Encountered an Error: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    # Step 1: Try the lightweight ping to reset the timer
    if is_app_sleeping(URL):
        # Step 2: Smart fallback to Selenium only if the ping hints it's asleep
        wake_up_with_selenium(URL)
