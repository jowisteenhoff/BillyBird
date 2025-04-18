import streamlit_authenticator as stauth

wachtwoorden = ["geheim123"]
gehasht = stauth.Hasher(wachtwoorden).generate()
print(gehasht[0])
