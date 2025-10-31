from app.models import engine, Employee, Base
from sqlalchemy.orm import Session

try:
    # Test connection
    with engine.connect() as connection:
        print("✅ Database connected successfully!")
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    print("✅ Tables created successfully!")
    
    # Test creating a record
    with Session(engine) as session:
        test_employee = Employee(
            employee_id="TEST001",
            first_name="Test",
            last_name="User",
            email="test@example.com",
            gender="Male"
        )
        session.add(test_employee)
        session.commit()
        print("✅ Test record created successfully!")
        
        # Clean up test record
        session.delete(test_employee)
        session.commit()
        print("✅ Test record deleted successfully!")
        
    print("\n🎉 Database is ready! You can proceed to Step 5.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure PostgreSQL is running")
    print("2. Check your password in .env file")
    print("3. Ensure 'welltrackdb' database exists")