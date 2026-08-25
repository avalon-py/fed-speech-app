import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# Replace with your actual Streamlit URL
URL = "https://your-app-name.streamlit.app/"

# Set up headless Chrome options for GitHub Actions
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(options=chrome_options)

try:
    print(f"Visiting app: {URL}")
    driver.get(URL)
    time.sleep(5)  # Wait for page to load

    # Look for the specific "Yes, get this app back up!" button text
    buttons = driver.find_elements(By.TAG_NAME, "button")
    wake_button = None
    
    for btn in buttons:
        if "get this app back up" in btn.text.lower() or "wake" in btn.text.lower():
            wake_button = btn
            break

    if wake_button:
        print("App is asleep! Clicking the Wake Up button...")
        wake_button.click()
        time.sleep(15)  # Give it time to spin up the container
        print("Wake up command sent successfully.")
    else:
        print("App is already awake and running smoothly!")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    driver.quit()
