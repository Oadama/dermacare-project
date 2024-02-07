import streamlit as st
from st_pages import Page, show_pages, add_page_title

add_page_title('FAQ')

st.markdown("**Dermacare peut-il remplacer un médecin ?**")
st.write("Non, Dermacare ne réalise que des prédictions et ne remplace en aucun cas une visite médicale.")


st.markdown("**J'ai un résultat préoccupant, que dois- faire ?**")
st.write("Nous vous recommandons de prendre rendez-vous avec un médecin pour qu'il analyse votre lésion. Les résultats du site n'ont qu'une valeur prédictive.")

st.markdown('**Comment prendre la photo ?**')
st.write('La photo doit être centrée sur la lésion cutanée.')



hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
