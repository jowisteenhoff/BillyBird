
# --------------------------------------------------
#  Benodigden pakketen 
# --------------------------------------------------
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from xgboost import XGBRegressor
import altair as alt

# -------------------------------------
#  Pagina instellingen
# --------------------------------------------------
# HTML Styling voor centreren van de hele pagina
st.markdown("<style>body {text-align: center;}</style>", unsafe_allow_html=True)

# Zet de favicon
st.markdown(
    """
    <head>
        <link rel="icon" href="logo_bird.ico" type="image/x-icon">
    </head>
    """, 
    unsafe_allow_html=True
)

st.markdown(
    """
    <script>
    const sidebar = window.parent.document.querySelector('section[data-testid="stSidebar"]');
    if (sidebar) {
        sidebar.style.transform = 'translateX(-100%)';
    }
    </script>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
#  Data laden
# --------------------------------------------------
@st.cache_data
def laad_excel_bestand(pad):
    return pd.read_excel(pad) 


# --------------------------------------------------
#  Authenticatie instellen
# --------------------------------------------------
# Configuratiebestand inladen
with open("Autenticator/config.yaml") as bestand:
    config = yaml.load(bestand, Loader=SafeLoader)

# Authenticatie-object aanmaken
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# Check of gebruiker al is ingelogd
status = st.session_state.get("authentication_status")

# Logo boven loginformulier (alleen als nog niet ingelogd)
if status is None:
    kolom_links, kolom_midden, kolom_rechts = st.columns([1, 7, 1])
    with kolom_midden:
        st.image("Logo/logo_textonly.png")
        st.markdown("<br>", unsafe_allow_html=True)

# Loginformulier tonen
login_result = authenticator.login(
    location="main",
    fields={
        "Form name": "Inloggen",
        "Username": "Gebruikersnaam",
        "Password": "Wachtwoord",
        "Login": "Inloggen"
    }
)

# Loginstatus ophalen
if login_result is not None:
    naam, status, gebruikersnaam = login_result
else:
    naam = st.session_state.get("name")
    status = st.session_state.get("authentication_status")
    gebruikersnaam = st.session_state.get("username")

# --------------------------------------------------
# Loginstatus verwerken
# --------------------------------------------------

if status is None:
    st.info("Voer je gebruikersnaam en wachtwoord in om toegang te krijgen.")

elif status is False:
    st.error("De gebruikersnaam of het wachtwoord is onjuist.")

elif status is True:
    authenticator.logout("Uitloggen", "sidebar")
    # --------------------------------------------------
    # PROGNOSE MODEL
    # --------------------------------------------------
    # --- Data inladen ---
    leden_pad = "Test data/hist_leden_dag.xlsx"
    tickets_pad = "Test data/hist_tickets_dag.xlsx"

    try:
        df_leden = laad_excel_bestand(leden_pad)
        df_tickets = laad_excel_bestand(tickets_pad)
        st.success(f'Welkom {naam}, je bent succesvol ingelogd en de data is succesvol ingeladen!')
    except Exception as e:
        st.error(f'Welkom {naam}, je bent succesvol ingelogd, maar er gaat iets fout bij inladen van bestanden: {e}')
        st.stop()

    # --- Sidebar: datumselectie ---
    st.sidebar.header("Selecteer startdatum")
    st.sidebar.write("Kies een datum voor een fictieve analyse om de kracht van het model te laten zien")
    jaren = list(range(2022, 2026))
    maanden = list(range(1, 13))
    dagen = list(range(1, 32))

    jaar = st.sidebar.selectbox("Jaar", jaren, index=2)
    maand = st.sidebar.selectbox("Maand", maanden, index=datetime.now().month - 1)
    dag = st.sidebar.selectbox("Dag", dagen, index=datetime.now().day - 1)

    try:
        startdatum = datetime(jaar, maand, dag)
        st.sidebar.success(f"Gekozen startdatum: {startdatum.strftime('%Y-%m-%d')}")
    except ValueError:
        st.sidebar.error("Ongeldige datumcombinatie.")
        st.stop()


    # Zorg dat de datumkolommen datetime-format zijn
    df_leden['date'] = pd.to_datetime(df_leden['date'])
    df_tickets['date'] = pd.to_datetime(df_tickets['date'])

    # --- Split naar train/test sets ---
    test_dagen = [startdatum + timedelta(days=i) for i in range(4)]

    test_leden = df_leden[df_leden['date'].isin(test_dagen)]
    train_leden = df_leden[~df_leden['date'].isin(test_dagen)]

    test_tickets = df_tickets[df_tickets['date'].isin(test_dagen)]
    train_tickets = df_tickets[~df_tickets['date'].isin(test_dagen)]

    # --- Voorspelling ---
    features = [
        'rolling_mean_3dagen', 'rolling_vakantie_interactie', 'temperatuur',
        'temperatuur_weekend_interactie', 'vakantie_bin', 'rolling_max_3dagen',
        'weekend_bin', 'weekdag', 'event_weer_interactie', 'regen_bin', 'schooldag_bin'
    ]

    def gewicht_custom(x):
        return np.log1p(x)

    # Sample weights
    sample_weights_leden = gewicht_custom(train_leden['Tickettotaal'])
    sample_weights_tickets = gewicht_custom(train_tickets['Tickettotaal'])

    @st.cache_resource 
    def run_model(train_df, test_df, sample_weights, label):
        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.1,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=1.0,
            objective='reg:absoluteerror',
            random_state=42
        )
        model.fit(train_df[features], train_df['Tickettotaal'], sample_weight=sample_weights)
        preds = model.predict(test_df[features])
        return pd.DataFrame({
            'date': test_df['date'].values,
            'werkelijk': test_df['Tickettotaal'].values,
            'voorspeld': preds
        })

    result_tickets = run_model(train_tickets, test_tickets, sample_weights_tickets, "Tickets")
    result_leden = run_model(train_leden, test_leden, sample_weights_leden, "Leden")

    # --- Totaal morgen & delta berekening ---
    gisteren = startdatum
    morgen = startdatum + timedelta(days=1)

    # Huidige voorspellingen
    voorspeld_leden_morgen = result_leden[result_leden['date'] == morgen]['voorspeld'].values[0]
    voorspeld_tickets_morgen = result_tickets[result_tickets['date'] == morgen]['voorspeld'].values[0]
    voorspeld_leden_gisteren = result_leden[result_leden['date'] == gisteren]['voorspeld'].values[0]
    voorspeld_tickets_gisteren = result_tickets[result_tickets['date'] == gisteren]['voorspeld'].values[0]

    totaal_morgen = round(voorspeld_leden_morgen + voorspeld_tickets_morgen)
    totaal_gisteren = round(voorspeld_leden_gisteren + voorspeld_tickets_gisteren)
    delta = totaal_morgen - totaal_gisteren

     # Emojis voor weertype
    weertype_emoji = {
        'bovenlucht onzichtbaar': '☁️',
        'lichte motregen': '🌧️',
        'helder': '☀️',
        'bewolkt': '🌥️',
        'zwaar bewolkt': '☁️',
        'gedeeltelijk bewolkt': '🌥️',
        'lichte regen': '🌧️',
        'licht bewolkt': '🌤️',
        'matige of zware regen met onweer': '⛈️',
        'lichte sneeuw': '❄️',
        'matige regen': '🌧️',
        'mist': '🌫️',
        'ijzige mist': '❄️🌫️',
        'onweer mogelijk': '⚡'
    }

    # Voor morgen de temperatuur en weertype ophalen
    weertype_morgen = df_leden[df_leden['date'] == morgen]['weertype_meestvoorkomen'].values[0]
    temp_morgen = df_leden[df_leden['date'] == morgen]['temperatuur'].mean()  
    # Gebruik de emoji voor het weertype van morgen
    weertype_emoji_morgen = weertype_emoji.get(weertype_morgen, '🌤️')  # Standaard 'licht bewolkt' als geen match

    # --------------------------------------------------
    # WEERGAVE IN DASHBOARD (OPENING + MORGEN)
    # --------------------------------------------------
    st.image("logo_textonly.png")
    st.title("_Bezoekersdashboard_")
    st.markdown("Gebruik de selectie in de zijbalk om een startdatum te kiezen voor voorspellingen.")
    st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader(f"Verwachte bezoekers morgen _({morgen.strftime('%d %B')})_")
    st.markdown(f"<h1 style='font-size: 55px; color: #4CAF50;'>{totaal_morgen:.0f}</h1>", unsafe_allow_html=True)
    
    # Leden en geen leden
    delta_leden = round(voorspeld_leden_morgen - voorspeld_leden_gisteren)
    delta_tickets = round(voorspeld_tickets_morgen - voorspeld_tickets_gisteren)
    col1, col2, col3, col4, col5 = st.columns([3,3,0.5,3,3])
    col2.metric(
        "Waarvan leden-entree:", 
        f"{voorspeld_leden_morgen:.0f}"
    )
    col4.metric(
        "Waarvan normale-entree:", 
        f"{voorspeld_tickets_morgen:.0f}"
    )

    # Vakantie- en evenementen-informatie voor morgen
    is_vakantie = df_leden[df_leden['date'] == morgen]['vakantie_bin'].values[0]  # Check of het vakantie is
    vakantie_naam = df_leden[df_leden['date'] == morgen]['vakantietype'].values[0]  # De naam van de vakantie (als beschikbaar)
    vakantie_text = vakantie_naam if is_vakantie == 1 else "Geen vakantie"
    
    event_row = df_leden[df_leden['date'] == morgen]
    if not event_row.empty:
        is_event = event_row['event_weer_interactie'].values[0]
        event_naam = event_row['event'].values[0]
        
        if pd.isna(event_naam) or is_event in [0, 0.0, False]:
            event_text = "Geen evenement"
        else:
            event_text = str(event_naam)
    else:
        event_text = "Geen data"

    # Weergave van de voorspelling voor morgen
    #st.markdown(f"**Weertype:** {weertype_emoji_morgen} **Temperatuur:** {temp_morgen:.1f}°C")
    #st.write(f"{vakantie_text}")
    #st.write(f"{event_text}")

    # --------------------------------------------------
    # WEERGAVE IN DASHBOARD (PER DRIE DAGEN)
    # --------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Bezoekersaantallen komende dagen")

    # --- Bereken de gemiddelde temperatuur voor de komende dagen ---
    dagkenmerken = [startdatum + timedelta(days=i) for i in range(1, 4)]  # Overmorgen, Dag 3, Dag 4
    temperaturen = []
    weer_types = []

    # Vertaal de dagen van de week naar Nederlands
    dag_nl = {
        'Monday': 'Maandag',
        'Tuesday': 'Dinsdag',
        'Wednesday': 'Woensdag',
        'Thursday': 'Donderdag',
        'Friday': 'Vrijdag',
        'Saturday': 'Zaterdag',
        'Sunday': 'Zondag'}

    #Bepalen weertypes
    for dag in dagkenmerken:
        # Gemiddelde temperatuur en weertype per dag ophalen
        temp_dag = df_leden[df_leden['date'] == dag]['temperatuur'].mean()  # Gebruik temperatuur kolom
        if not pd.isna(temp_dag):
            temperaturen.append(round(temp_dag, 1))  # Voeg toe als temperatuur beschikbaar is
        else:
            temperaturen.append('Onbekend')  # Als geen temperatuur beschikbaar is
        
        # Weertype ophalen
        if not df_leden[df_leden['date'] == dag].empty:
            weertype = df_leden[df_leden['date'] == dag]['weertype_meestvoorkomen'].values[0]
            weer_types.append(weertype_emoji.get(weertype, '🌤️'))  # Gebruik emoji voor weertype
        else:
            weer_types.append('🌤️')  # Standaard als geen weertype is

    # --- Drie kolommen met de voorspelling per dag  loopen ---
    col1, col2, col3 = st.columns([1, 1, 1])  # Gelijken verhouding van kolommen

    for i, dag in enumerate(dagkenmerken):
        with [col1, col2, col3][i]:
            # Dag van de week in het Engels
            dag_naam_english = dag.strftime('%A')
            dag_naam_nl = dag_nl.get(dag_naam_english, dag_naam_english)  # Vertaal naar Nederlands
            
            # Datum in de vorm van '5 juni'
            datum_nl = dag.strftime('%d %B')  # Bijvoorbeeld '5 juni'
            
            # Voorspelde bezoekers (leden + tickets)
            voorspeld_leden = result_leden[result_leden['date'] == dag]['voorspeld'].values[0]
            voorspeld_tickets = result_tickets[result_tickets['date'] == dag]['voorspeld'].values[0]
            totaal = voorspeld_leden + voorspeld_tickets
            
            # Temperatuur en Weertype
            temp = temperaturen[i]
            weertype = weer_types[i]

            # Vakantie- en evenementen-informatie voor de dag
            is_vakantie = df_leden[df_leden['date'] == dag]['vakantie_bin'].values[0]  # Check of het vakantie is
            is_event = df_leden[df_leden['date'] == dag]['event_weer_interactie'].values[0]  # Check of er een evenement is

            vakantie_naam = df_leden[df_leden['date'] == dag]['vakantietype'].values[0]  # De naam van de vakantie (als beschikbaar)
            event_naam = df_leden[df_leden['date'] == dag]['event'].values[0]  # De naam van het evenement (als beschikbaar)

            # Bepalen van de tekst om weer te geven
            vakantie_text = vakantie_naam if is_vakantie == 1 else "Geen vakantie"
            event_text = event_naam if is_event == 1 else "Geen evenement"

            # Datum en dag van de week
            st.markdown(f"**_{dag_naam_nl} {datum_nl}_**")  

            # Weer 2 
            st.markdown(f"<h1 style='text-align: center; font-size: 120px;'>{weertype}</h1>", unsafe_allow_html=True)
            st.markdown(f"# **{temp}°C**", unsafe_allow_html=True)

            # Leden en Tickets met st.write()
            st.markdown("---")
            st.write(f"**Totaal aantal bezoekers:**")
            st.markdown(f"## {int(totaal)}")
            st.markdown("---")
            st.write(f"Verwacht leden-entrees:")
            st.markdown(f"###  {int(voorspeld_leden)}")
            st.markdown("---")
            st.write(f"Verwacht tickets-entrees:")
            st.markdown(f"### {int(voorspeld_tickets)}")
            st.markdown("---")
            st.write(f"{vakantie_text}")
            st.write(f"{event_text}")
   
    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------
    # WEERGAVE IN DASHBOARD (GRAFIEK)
    # --------------------------------------------------
    # Combineer historische + voorspelde data
    verleden_dagen = [startdatum - timedelta(days=i) for i in range(4, 0, -1)]
    toekomst_dagen = [startdatum + timedelta(days=i) for i in range(4)]

    # Historische data ophalen (werkelijke waarden)
    verleden_leden = df_leden[df_leden['date'].isin(verleden_dagen)][['date', 'Tickettotaal']].copy()
    verleden_tickets = df_tickets[df_tickets['date'].isin(verleden_dagen)][['date', 'Tickettotaal']].copy()
    verleden_leden['type'] = 'Leden'
    verleden_tickets['type'] = 'Tickets'

    # Voorspellingen ophalen
    voorspeld_leden = result_leden[result_leden['date'].isin(toekomst_dagen)][['date', 'voorspeld']].copy()
    voorspeld_tickets = result_tickets[result_tickets['date'].isin(toekomst_dagen)][['date', 'voorspeld']].copy()
    voorspeld_leden = voorspeld_leden.rename(columns={'voorspeld': 'Tickettotaal'})
    voorspeld_tickets = voorspeld_tickets.rename(columns={'voorspeld': 'Tickettotaal'})
    voorspeld_leden['type'] = 'Leden'
    voorspeld_tickets['type'] = 'Tickets'

    # Combineer alles in één dataframe
    grafiek_df = pd.concat([verleden_leden, verleden_tickets, voorspeld_leden, voorspeld_tickets])
    grafiek_df['bron'] = ['Historisch'] * (len(verleden_leden) + len(verleden_tickets)) + ['Voorspelling'] * (len(voorspeld_leden) + len(voorspeld_tickets))

    #Kleuren
    kleurenschaal = alt.Scale(
        domain=['Leden', 'Tickets'],
        range=['#66C2A5', '#FC8D62']  # groen voor leden, oranje voor tickets
    )
    grijstinten = alt.Scale(
        domain=['Leden', 'Tickets'],
        range=['#C0C0C0', '#808080']
    )

    chart_hist = alt.Chart(grafiek_df[grafiek_df['bron'] == 'Historisch']).mark_bar(size=20).encode(
    x=alt.X('date:T', title='Datum', axis=alt.Axis(format='%d-%b', tickCount='day')),
    y=alt.Y('Tickettotaal:Q', title='Aantal bezoekers'),
    color=alt.Color('type:N', scale=grijstinten, legend=None),
    tooltip=['date:T', 'type:N', 'Tickettotaal:Q']
    )

    chart_voorsp = alt.Chart(grafiek_df[grafiek_df['bron'] == 'Voorspelling']).mark_bar(size=20).encode(
        x=alt.X('date:T', title='Datum', axis=alt.Axis(format='%d-%b', tickCount='day')),
        y=alt.Y('Tickettotaal:Q', title='Aantal bezoekers'),
        color=alt.Color('type:N', scale=kleurenschaal, legend=alt.Legend(orient='bottom')),
        tooltip=['date:T', 'type:N', 'Tickettotaal:Q']
    )

    combi = alt.layer(chart_hist, chart_voorsp).resolve_scale(color='independent')
    st.altair_chart(combi.properties(width=800, height=400), use_container_width=True)

    # --------------------------------------------------
    # PERSOONLIJKE GEGEVEN WIJZIGEN
    # --------------------------------------------------
    with st.expander("Persoonlijke instellingen"):
        try:
            if authenticator.update_user_details(
                gebruikersnaam,
                fields={
                    "Form name": "Persoonsgegevens wijzigen",
                    "Field": "Te wijzigen veld",
                    "First name": "Voornaam",
                    "Last name": "Achternaam",
                    "Email": "E-mailadres",
                    "New value": "Nieuwe waarde",
                    "Update": "Bijwerken"
                }
            ):
                st.success("Je gegevens zijn succesvol bijgewerkt.")
        except Exception as fout:
            st.error(f"Er is iets misgegaan bij het bijwerken van je gegevens: {fout}")

        try:
            wachtwoord_success = authenticator.reset_password(
                gebruikersnaam,
                fields={
                    "Form name": "Wachtwoord wijzigen",
                    "Current password": "Huidig wachtwoord",
                    "New password": "Nieuw wachtwoord",
                    "Repeat password": "Herhaal wachtwoord",
                    "Reset": "Wijzigen"
                }
            )
            if wachtwoord_success:
                st.success("Je wachtwoord is succesvol gewijzigd.")
            elif "Reset password" in st.session_state and st.session_state["Reset password"]:
                st.warning("Het wijzigen van je wachtwoord is niet gelukt. Controleer je invoer.")
        except Exception as fout:
            st.error(f"Er is een fout opgetreden bij het wijzigen van je wachtwoord: {fout}")

    # --------------------------------------------------
    # WERKELIJKE RESULTATEN
    # --------------------------------------------------
    st.markdown("---")
    st.markdown("<br><br><br>", unsafe_allow_html=True)

    st.subheader("Werkelijk resultaten voor de komende 4 dagen")
    st.write("Voorspellingen - Tickets")
    st.dataframe(result_tickets.round(1))
    st.write("Voorspellingen - Leden")
    st.dataframe(result_leden.round(1))


    st.subheader("Train/Test splitsing")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Testset - Leden**")
        st.dataframe(test_leden)
        st.markdown("**Trainset - Leden**")
        st.dataframe(train_leden)
    with col2:
        st.markdown("**Testset - Tickets**")
        st.dataframe(test_tickets)
        st.markdown("**Trainset - Tickets**")
        st.dataframe(train_tickets)

    
 





        colA, colB, colC, colD = st.columns([3.3,1,1,3])  
        with colB:
                # Weergeef het weertype (emoji) met grotere grootte
                st.markdown(f"<h1 style='text-align: center; font-size: 100px;'>{weertype_emoji_morgen}</h1>", unsafe_allow_html=True)

        with colC:
                # Weergeef de temperatuur met grotere grootte
                if not pd.isna(temp_morgen):
                    st.markdown(f"<h1 style='text-align: center; font-size: 30px;'>{temp_morgen:.1f}°C</h1>", unsafe_allow_html=True)
                else:
                    st.markdown("<h1 style='text-align: center; font-size: 30px;'>Onbekend</h1>", unsafe_allow_html=True)

