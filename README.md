# dc-langchain-tutorial
Langchain Tutorial from DataCamp
https://campus.datacamp.com/courses/developing-llm-applications-with-langchain/


#create virtual environment and install pip and dotenv
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install --upgrade pip
$ pip install python-dotenv

#Install necessary modules for this project
$ pip install langchain langchain-huggingface transformers torch accelerate
$ pip freeze > requirements.txt

copy utils/AppConfig.py from any python project
copy .gitignore from ~/poc/python/learning-python

Now open the project in pyCharm
Go to terminal and type which python
Output will show global installation folder
Open AppConfig.py
Click on link "Configure Python Interpreter"
Click "Add New Interpreter"--> "Add local interpreter"
Choose radio button "Existing" and click Ok

Go to terminal and type which python
Output will show local installation folder
command prompt will have "(.venv)" at the beginning

 
