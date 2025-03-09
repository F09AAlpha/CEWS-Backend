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

## Creating a Django Project
To start a new Django project, run:
```sh
django-admin startproject myproject
```

Replace `myproject` with your desired project name.

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

## Additional Dependencies
To install additional dependencies, use:
```sh
pip install -r requirements.txt
```
To generate a `requirements.txt` file with installed dependencies:
```sh
pip freeze > requirements.txt
