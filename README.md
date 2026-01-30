## Real-Time Audio Event Classification (Speech / Noise / Silence)

Low-latency audio classification pipeline using MFCC features + a lightweight CNN.
Built for wearable/headphone-style constraints (fixed sample rate, small model, fast inference).

### Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
