# Alpha-Backend

## Prerequisites
- Ensure you have **Python 3.12.6** installed. You can check your Python version by running:
  ```sh
  python --version
  ```
  or
  ```shA
  python3 --version
  ```

## Setting Up a Virtual Environment
It is recommended to use a virtual environment to manage dependencies.

### 1. Create a Virtual Environment
Run the following command in your project directory:
```sh
python -m venv venv
```

### 2. Activate the Virtual Environment
- **Windows**:
  ```sh
  venv\Scripts\activate
  ```
- **Mac/Linux**:
  ```sh
  source venv/bin/activate
  ```

## Installing Django
Once the virtual environment is activated, install Django using pip:
```sh
pip install django
```

Verify the installation:
```sh
python -m django --version
```

## Running the Django Server
Navigate to your project directory:
```sh
cd myproject
```
Run the development server:
```sh
python manage.py runserver
```
Your Django project should now be accessible at [http://127.0.0.1:8000/](http://127.0.0.1:8000/).

## Deactivating the Virtual Environment
To exit the virtual environment, simply run:
```sh
deactivate
```
## Running with Docker

### Prerequisites
- Docker and Docker Compose installed on your system

### Starting the Application
Run the following command:
```sh
./docker-start.sh
```
Or manually with:
```sh
docker-compose up --build
```
Once running, the application will be available at:
- http://localhost:8000

### Stopping the Application
Press ```Ctrl+C``` in the terminal where docker-compose is running, or run:
```sh
docker-compose down
```

### Database Setup
Environment Configuration
- Create a .env file in the project root using the .env.example as a template. Reach out to your scrum master for Database credentials

## Running Database Migrations
Update the .env file with your RDS credentials

- After connecting to the database, run migrations to create the schema:
With Docker:

create migrations:
```sh
docker-compose run web python manage.py makemigrations
```
apply the migrations: 
```sh
docker-compose run web python manage.py migrate
```
Without Docker (with virtual environment activated):

create migrations:
```sh
python manage.py makemigrations
```
apply the migrations: 
```sh
python manage.py migrate
```

## Additional Dependencies
To install additional dependencies, use:
```sh
pip install -r requirements.txt
```
To generate a `requirements.txt` file with installed dependencies:
```sh
pip freeze > requirements.txt