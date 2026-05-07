import sqlite3
import os

def init_db():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_dir = os.path.join(base_dir, "data")
    db_path = os.path.join(data_dir, "hr_db.sqlite")
    
    os.makedirs(data_dir, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create employees table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS employees (
        id TEXT PRIMARY KEY,
        name TEXT,
        annual_leave_balance INTEGER,
        sick_leave_balance INTEGER
    )
    ''')
    
    # Create leave_requests table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS leave_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id TEXT,
        start_date TEXT,
        end_date TEXT,
        leave_type TEXT,
        days_count INTEGER,
        FOREIGN KEY(employee_id) REFERENCES employees(id)
    )
    ''')

    # Create travel_requests table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS travel_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id TEXT,
        source TEXT,
        destination TEXT,
        outbound_date TEXT,
        return_date TEXT,
        is_round_trip BOOLEAN,
        selected_flight TEXT,
        status TEXT,
        FOREIGN KEY(employee_id) REFERENCES employees(id)
    )
    ''')
    
    # Seed dummy data
    employees = [
        ('123', 'ABC', 20, 10),
        ('EMP001', 'Ravi Kumar', 25, 12),
        ('EMP002', 'Ashish Sharma', 15, 8),
        ('EMP003', 'Smita Patil', 20, 10),
        ('EMP004', 'Prerna Singh', 10, 5),
        ('EMP005', 'Pravin Jadhav', 30, 15)
    ]
    
    cursor.executemany("INSERT OR IGNORE INTO employees (id, name, annual_leave_balance, sick_leave_balance) VALUES (?, ?, ?, ?)",
                       employees)
    
    conn.commit()
    conn.close()
    print(f"Database initialized at {db_path}")

if __name__ == "__main__":
    init_db()
