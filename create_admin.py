import sys
from main import SessionLocal, AdminUser, get_password_hash

def create_admin(username, password):
    db = SessionLocal()
    existing = db.query(AdminUser).filter(AdminUser.username == username).first()
    if existing:
        print(f"User '{username}' already exists. Use a different username.")
        return
    
    hashed_pw = get_password_hash(password)
    admin = AdminUser(username=username, hashed_password=hashed_pw)
    db.add(admin)
    db.commit()
    print(f"\n✅ Successfully created admin user: '{username}'")
    print("You can now login to the dashboard using these credentials.")
    db.close()

if __name__ == "__main__":
    print("=== Create a New Admin User ===")
    user = input("Enter admin username: ").strip()
    pw = input("Enter admin password: ").strip()
    if user and pw:
        create_admin(user, pw)
    else:
        print("Username and password are required.")
