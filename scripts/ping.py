import sys
import urllib.request
import urllib.parse
import http.cookiejar
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# Replace with your actual Streamlit URL
URL = "https://streamlit.app"

def is_app_sleeping(url):
    """Performs an HTTP request with cookie handling to clear 303 loops."""
    try:
        # Build a cookie handler to satisfy Streamlit's routing redirects
        cj = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
        
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        
        with opener.open(req, timeout=15) as response:
            html = response.read().decode('utf-8', errors='ignore')
            
            # Smart check for Streamlit sleep page indicators
            if "get this app back up" in html.lower() or "wake" in html.lower():
                print("⚠️ App appears to be showing the sleep screen.")
                return True
                
            print(f"✅ Simple ping successful! Status: {response.getcode()}. Timer reset.")
            return False
            
    except Exception as e:
        print(f"⚠️ Simple ping failed or looped ({e}). Resorting to browser check...")
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
        time.sleep(6)  # Wait for structural JS rendering

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
            time.sleep(15) 
            print("🎉 Wake up signal sent successfully.")
        else:
            print("😎 App is verified AWAKE in browser view. No action required.")

    except Exception as e:
        print(f"❌ Selenium Encountered an Error: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    if is_app_sleeping(URL):
        wake_up_with_selenium(URL)
