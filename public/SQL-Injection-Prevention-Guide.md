# 🔐 SQL Injection Prevention Guide

## What is SQL Injection?

SQL Injection is a code injection technique that exploits security vulnerabilities in an application's database layer. Attackers can inject malicious SQL code into queries, potentially allowing them to:

- Access unauthorized data
- Modify or delete data
- Execute administrative operations on the database
- Issue commands to the operating system

---

## 🚨 Vulnerable Code Examples

### Example 1: String Concatenation (Python)

**❌ VULNERABLE - DO NOT USE**

```python
# Dangerous: User input directly concatenated into SQL query
username = request.form['username']
password = request.form['password']

query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
cursor.execute(query)
```

**Attack Example:**
```
Username: admin' --
Password: anything
```

This results in:
```sql
SELECT * FROM users WHERE username = 'admin' --' AND password = 'anything'
```

The `--` comments out the rest of the query, bypassing password check!

---

### Example 2: String Concatenation (JavaScript/Node.js)

**❌ VULNERABLE - DO NOT USE**

```javascript
const userId = req.params.id;
const query = "SELECT * FROM users WHERE id = " + userId;

connection.query(query, (error, results) => {
  // Process results
});
```

**Attack Example:**
```
/users/1 OR 1=1
```

This results in:
```sql
SELECT * FROM users WHERE id = 1 OR 1=1
```

Returns all users instead of just one!

---

### Example 3: Dynamic WHERE Clause (PHP)

**❌ VULNERABLE - DO NOT USE**

```php
$searchTerm = $_GET['search'];
$sql = "SELECT * FROM products WHERE name LIKE '%" . $searchTerm . "%'";
$result = mysqli_query($conn, $sql);
```

**Attack Example:**
```
search=%' OR '1'='1
```

This results in:
```sql
SELECT * FROM products WHERE name LIKE '%%' OR '1'='1%'
```

Returns all products!

---

## ✅ Secure Code Examples

### Example 1: Parameterized Queries (Python)

**✅ SECURE - USE THIS**

```python
# Safe: Using parameterized query with placeholders
username = request.form['username']
password = request.form['password']

query = "SELECT * FROM users WHERE username = %s AND password = %s"
cursor.execute(query, (username, password))
```

**Why it's safe:** The database driver properly escapes and handles user input as data, not as SQL code.

---

### Example 2: Prepared Statements (JavaScript/Node.js with MySQL)

**✅ SECURE - USE THIS**

```javascript
const userId = req.params.id;
const query = "SELECT * FROM users WHERE id = ?";

connection.query(query, [userId], (error, results) => {
  // Process results
});
```

**Why it's safe:** The `?` placeholder is replaced safely by the database driver.

---

### Example 3: ORM Usage (Python with SQLAlchemy)

**✅ SECURE - USE THIS**

```python
# Using ORM (Object-Relational Mapping)
from sqlalchemy import select
from models import User

username = request.form['username']
password = request.form['password']

stmt = select(User).where(User.username == username, User.password == password)
result = session.execute(stmt)
user = result.scalar_one_or_none()
```

**Why it's safe:** ORMs automatically handle parameterization and escaping.

---

### Example 4: Prepared Statements (PHP with PDO)

**✅ SECURE - USE THIS**

```php
$searchTerm = $_GET['search'];
$sql = "SELECT * FROM products WHERE name LIKE :search";
$stmt = $pdo->prepare($sql);
$stmt->execute(['search' => "%{$searchTerm}%"]);
$results = $stmt->fetchAll();
```

**Why it's safe:** PDO prepared statements separate SQL structure from data.

---

## 🛡️ Best Practices

### 1. **Always Use Parameterized Queries or Prepared Statements**
   - Never concatenate user input into SQL queries
   - Use placeholders (`?`, `:param`, `%s`, etc.)

### 2. **Use ORM Libraries**
   - SQLAlchemy (Python)
   - Sequelize (Node.js)
   - Entity Framework (C#)
   - Hibernate (Java)
   - Eloquent (PHP/Laravel)

### 3. **Input Validation**
   - Validate data types (e.g., ensure ID is numeric)
   - Whitelist acceptable values when possible
   - Limit input length
   - **Note:** Validation is defense-in-depth, NOT a replacement for parameterized queries

### 4. **Principle of Least Privilege**
   - Database users should have minimal permissions
   - Application should not use admin/root database accounts
   - Use separate accounts for read vs. write operations

### 5. **Error Handling**
   - Don't expose detailed database errors to users
   - Log errors securely for debugging
   - Return generic error messages to users

### 6. **Additional Security Layers**
   - Use Web Application Firewalls (WAF)
   - Implement rate limiting
   - Regular security audits and penetration testing
   - Keep database software updated

---

## 🔍 Real-World SQL Injection Patterns

### Login Bypass

**Attack:**
```
Username: ' OR '1'='1' --
Password: anything
```

**Vulnerable Query:**
```sql
SELECT * FROM users WHERE username = '' OR '1'='1' --' AND password = 'anything'
```

### Data Extraction

**Attack:**
```
productId=1 UNION SELECT username, password, NULL FROM users --
```

**Vulnerable Query:**
```sql
SELECT name, price, description FROM products WHERE id = 1 UNION SELECT username, password, NULL FROM users --
```

### Database Manipulation

**Attack:**
```
userId=1; DROP TABLE users; --
```

**Vulnerable Query (if multiple statements allowed):**
```sql
SELECT * FROM users WHERE id = 1; DROP TABLE users; --
```

---

## 🧪 Testing for SQL Injection

### Common Test Inputs:
- `'` (single quote) - causes syntax error if vulnerable
- `' OR '1'='1` - common bypass attempt
- `'; DROP TABLE users; --` - destructive test
- `1' UNION SELECT NULL, NULL --` - data extraction test

### Automated Tools:
- **SQLMap** - Automated SQL injection detection and exploitation
- **OWASP ZAP** - Web application security scanner
- **Burp Suite** - Professional security testing

---

## 📚 Additional Resources

- [OWASP SQL Injection Prevention Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html)
- [OWASP Top 10 - Injection](https://owasp.org/www-project-top-ten/)
- [CWE-89: SQL Injection](https://cwe.mitre.org/data/definitions/89.html)
- [SQL Injection on PortSwigger](https://portswigger.net/web-security/sql-injection)

---

## 🎯 Quick Reference

| Language/Framework | Secure Method |
|-------------------|---------------|
| Python (SQLite) | `cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))` |
| Python (PostgreSQL) | `cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))` |
| Node.js (MySQL) | `connection.query("SELECT * FROM users WHERE id = ?", [userId])` |
| PHP (PDO) | `$stmt = $pdo->prepare("SELECT * FROM users WHERE id = :id"); $stmt->execute(['id' => $userId]);` |
| Java (JDBC) | `PreparedStatement pstmt = con.prepareStatement("SELECT * FROM users WHERE id = ?"); pstmt.setInt(1, userId);` |
| C# (ADO.NET) | `SqlCommand cmd = new SqlCommand("SELECT * FROM users WHERE id = @id", conn); cmd.Parameters.AddWithValue("@id", userId);` |

---

**Remember:** The only reliable defense against SQL injection is to use parameterized queries or prepared statements. Input validation alone is not sufficient!
