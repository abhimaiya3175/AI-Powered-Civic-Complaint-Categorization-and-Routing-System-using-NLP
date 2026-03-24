from main import SessionLocal, Complaint

try:
    db = SessionLocal()
    comp = Complaint(
        audio_path="uploads/mock_test.wav",
        original_text="There is a huge garbage dump near the park in Indiranagar.",
        translated_text="There is a huge garbage dump near the park in Indiranagar.",
        language="en",
        category="Solid Waste",
        location="Indiranagar",
        status="pending"
    )
    db.add(comp)
    db.commit()
    print("Mock complaint inserted successfully!")
except Exception as e:
    print(f"Error inserting mock data: {e}")
