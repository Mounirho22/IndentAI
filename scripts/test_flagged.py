import requests

res = requests.post("http://127.0.0.1:8000/process-lesson", json={
    "unit_name": "Unit 4: Loops",
    "topic": "Loops",
    "objective": "learn stuff"
})
print("POST response:", res.status_code, res.text)
