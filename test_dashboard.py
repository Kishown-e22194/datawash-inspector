import sys
import os

# Add the local source code path so it imports the latest uninstalled code
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from datawash import DataPipeline

print("Initializing DataPipeline...")
pipe = DataPipeline()

print("Loading data...")
pipe.load_data(r'C:\Users\sasee\OneDrive\Desktop\games.csv')

print("Generating dashboard locally...")
pipe.show_dashboard()

print("Testing Colab injection string generation...")
try:
    import IPython
    from IPython.display import HTML
    
    # We spoof being in Colab just to see if the iframe generation crashes
    import sys
    sys.modules['google.colab'] = type('MockColab', (), {})()
    print("Spoofing Colab environment to test iframe generation...")
    # Because we mocked google.colab, show_dashboard will try to display via IPython
    pipe.show_dashboard()
    print("Colab iframe generation successful!")
except Exception as e:
    print(f"Colab generation failed: {e}")
