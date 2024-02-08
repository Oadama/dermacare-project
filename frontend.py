import streamlit as st
import requests
#from PIL import Image
from st_pages import Page, show_pages, add_page_title
from pathlib import Path


st.set_page_config(
    page_title="Votre assistant cutané",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",

)



url='https://dermacare-api-kpmfnijgja-ew.a.run.app/predict_cnn'

#pages = st.source_util.get_pages('app.py')
show_pages(
    [
        Page("frontend.py", "Lesion prediction", ":mag:"),
        Page("faq.py", "FAQ", ":question:"),
    ]
)


def add_logo():

 # background-image: url(logo.png); (dans le style)
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {

                background-repeat: no-repeat;
                padding-top: 120px;
                background-position: 20px 20px;
            }
            [data-testid="stSidebarNav"]::before {
                content: "My Company Name";
                margin-left: 20px;
                margin-top: 20px;
                font-size: 30px;
                position: relative;
                top: 100px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )




#st.title("Dermasaaj votre assistant cutané")
def add_logo():
    st.markdown(
        """
        <style>
            .font {

                background-repeat: no-repeat;
                padding-top: 20px;
                background-position: 20px 20px;
            }

        </style>
        """,
        unsafe_allow_html=True,
    )
add_logo()
#image = Image.open("/app/dermasaaj/streamlit/dermacare-logo.png") #Brand logo image (optional)
#st.sidebar.image("illlustr3.png", use_column_width=True)

#Create two columns with different width
col1, col2 = st.columns( [0.8, 0.12])
with col1:               # To display the header text using css style
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; font-weight: 700;color: #FF9633;}
    #colorbis #ff7412
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Dermacare votre assistant cutané</p>', unsafe_allow_html=True)

#with col2:               # To display brand logo
  #  st.image(image,  width=150)

age = st.number_input('Votre âge ?', min_value=0, max_value=100, step=1)


sex = st.radio('Votre sexe', ('Homme', 'Femme'),horizontal=True)


lesion = st.selectbox('Localisation de la lésion', ['bras','main','tête/nuque','torse latéral','jambe/pied','oral/génital','paume/plante','dos','torse','autre'])

st.set_option('deprecation.showfileUploaderEncoding', False)

###########################################

uploaded_file = st.file_uploader("Choisissez une photo de votre lésion", type=['png', 'jpg'] )


class_cut= {'AK':'Kératose actinique',
         'BCC':'Carcinome basocellulaire',
         'BKL':'Kératose/lésion bégnine',
         'DF':'Dermatofibromes',
         'MEL':'Mélanome',
         'NV': "Grain de beauté bénin",
         'SCC': 'Carcinome épidermoïde',
         'VASC': 'Lésion vasculaire'

         }

def speech(lesion):

    if lesion=='NV':
        st.write('Le modèle a prédit un grain de beauté bénin aussi appelé Naevus :grinning:')
        st.write(' Nous vous rappelons que cette prédiction ne remplace pas un diagnostic fait par un médecin.')
        st.write('Informations complémentaires: [https://fr.wikipedia.org/wiki/Grain_de_beauté](https://fr.wikipedia.org/wiki/Grain_de_beaut%C3%A9)')

    elif lesion=='MEL':
        st.write("Le mélanome est un cancer de la peau développé à partir de cellules appelées mélanocytes. Lorsqu'il est dépisté à un stade précoce, il peut être traité efficacement.")
        st.write("Cette prédiction ne remplace en aucun cas un diagnostic fait par un médecin. Nous vous recommandons d'aller consulter rapidement un médecin pour faire examiner votre lésion.")
        st.write('Prendre rendez-vous avec un dermatologue sur [Doctolib](https://www.doctolib.fr/dermatologue)')

    elif st.write=='BCC':
        st.write("Le carcinome basocellulaire est un cancer de la peau. Son évolution est généralement lente et son traitement très efficace.")
        st.write("Cette prédiction ne remplace en aucun cas un diagnostic fait par un médecin. Nous vous recommandons d'aller consulter rapidement un médecin pour faire examiner votre lésion.")
        st.write('Prendre rendez-vous avec un dermatologue sur [Doctolib](https://www.doctolib.fr/dermatologue)')

    elif st.write=='SCC':
        st.write("Le carcinome épidermoïde est un cancer de la peau. Dépisté à temps il répond bien aux traitements.")
        st.write("Cette prédiction ne remplace en aucun cas un diagnostic fait par un médecin. Nous vous recommandons d'aller consulter rapidement un médecin pour faire examiner votre lésion.")
        st.write('Prendre rendez-vous avec un dermatologue sur [Doctolib](https://www.doctolib.fr/dermatologue)')
    else:
        st.write("Cette prédiction ne remplace en aucun cas un diagnostic fait par un médecin.")
        st.write('Prendre rendez-vous avec un dermatologue sur [Doctolib](https://www.doctolib.fr/dermatologue)')



if uploaded_file is not None:



    #image = Image.open(uploaded_file)
    st.image(uploaded_file, width=400,caption='Lésion cutanée')
    with st.spinner("Merci de patienter..."):

        bytes_data = uploaded_file.getvalue()

        res = requests.post(url, files={'img': bytes_data})
        print(res.status_code)
        if res.status_code == 200:
            #st.write(res.content)
            response=res.json()
            #st.write(f"{round(response['proba'],3)} pourcent de chance d'être en bonne santé")
            #st.write(response['message'])
            try:
                splito=response['message'].split()
                diag=splito[0]
                proba=splito[1]
                st.markdown(f"**Prédiction:** {class_cut[diag]} - **Fiabilité:**  {proba}%")
                speech(str(diag))
            except KeyError:
                st.write(res.content)
                print(response)


        else:
            st.markdown("**Oops**, something went wrong 😓 Please try again.")
            print(res.status_code, res.content)





hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
