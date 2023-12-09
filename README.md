# Ask Sadhguru 

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oOJkErWRT3gNRAfG89wLk4Dijynthvwx?usp=sharing)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/desaibhargav/VR/AskSadhguru/)


Repository for Sadhguru video recommendation project for the NGO Isha Foundation. The aim is to provide a natural language interface to users to "ask" questions and be given specific excerpts from Sadhguru's YouTube videos, blogs, or podcasts that "answer" those questions. 

This project was created during the pandemic of 2020, which was an especially hard time, in the cause of mental well-being and stability. From burdening existential questions, to practical advice, this project was made to be a guiding light for anyone and everyone. 

This PoC project was done in collaboration with the Isha Foundation, and was handed over to their technical team for deployment on their official app. 


STATUS: [POC READY]

## DEMO

Find the demo of the project pilot [here](https://drive.google.com/file/d/1b1PHEWk8HjW5ysHbsPNL2oAw0COIsG8b/view?usp=sharing)


## ON-GOING TASKS:
1. backend/recommender.py -> refactor code to integrate podcast recommendations along with YouTube recommendations.
2. frontend/utils.py -> refactor code to integrate podcast recommendations along with YouTube recommendations. 
3. backend/clients/podcast_client.py -> clean up code in podcast_client.py


## TO DO:
1. Update README.md (with basic usage, documentation and purpose of the project).
2. Take it from POC to PROD using [`FastAPI` x `Streamlit`] for frontend and [`Google Cloud Platform` / `Amazon Web Services`] as deployment platform.
3. **ADD. COMMENTS. EVERYWHERE.**
