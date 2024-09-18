from werkzeug.security import generate_password_hash

actual_password = 'loulou'
method = 'pbkdf2:sha256:260000'

password_hash = generate_password_hash(actual_password, method=method)
print(password_hash)

