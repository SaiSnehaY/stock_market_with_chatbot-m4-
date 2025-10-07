document.addEventListener('DOMContentLoaded', () => {
    const loginHeader = document.getElementById('login-header');
    const signupHeader = document.getElementById('signup-header');
    const loginForm = document.getElementById('login-form');
    const signupForm = document.getElementById('signup-form');

    // Simple user storage (for demonstration, replace with actual backend in production)
    const users = JSON.parse(localStorage.getItem('users')) || {};

    loginHeader.addEventListener('click', () => {
        loginHeader.classList.add('active');
        signupHeader.classList.remove('active');
        loginForm.classList.remove('hidden');
        signupForm.classList.add('hidden');
    });

    signupHeader.addEventListener('click', () => {
        signupHeader.classList.add('active');
        loginHeader.classList.remove('active');
        signupForm.classList.remove('hidden');
        loginForm.classList.add('hidden');
    });

    signupForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const username = signupForm.querySelector('#signup-username').value;
        const password = signupForm.querySelector('#signup-password').value;
        const confirmPassword = signupForm.querySelector('#signup-confirm-password').value;

        if (password !== confirmPassword) {
            alert('Passwords do not match!');
            return;
        }

        if (users[username]) {
            alert('Username already exists!');
            return;
        }

        users[username] = password;
        localStorage.setItem('users', JSON.stringify(users));
        alert('Signup successful! You can now log in.');
        loginHeader.click(); // Switch to login form
    });

    loginForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const username = loginForm.querySelector('#login-username').value;
        const password = loginForm.querySelector('#login-password').value;

        if (users[username] && users[username] === password) {
            alert('Login successful!');
            localStorage.setItem('loggedInUser', username);
            window.location.href = 'dashboard.html'; // Redirect to dashboard
        } else {
            alert('Invalid username or password!');
        }
    });
});