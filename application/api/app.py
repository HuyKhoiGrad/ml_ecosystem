import streamlit as st
from datetime import datetime, timedelta

from application.api import utils

st.set_page_config(page_title="Consume prediction", page_icon="🚀", layout="wide")


area = ["DK1", "DK2"]

device_type = [
    111,
    112,
    119,
    121,
    122,
    123,
    130,
    211,
    212,
    215,
    220,
    310,
    320,
    330,
    340,
    350,
    360,
    370,
    381,
    382,
    390,
    410,
    421,
    422,
    431,
    432,
    433,
    441,
    442,
    443,
    444,
    445,
    446,
    447,
    450,
    461,
    462,
    999,
]


def main():
    if "select" not in st.session_state:
        st.session_state["select"] = False
    area_choice = st.sidebar.selectbox("Select area", options=area, index=0, key="area")
    device_type_choice = st.sidebar.selectbox(
        "Select area", options=device_type, index=0, key="device"
    )
    number_last_hour = st.sidebar.text_input("Last hours")
    number_next_hour = st.sidebar.text_input("Next hours")
    if st.sidebar.button("Run"):
        st.session_state["select"] = True
    if (
        st.session_state["select"] == True
        and area_choice is not None
        and device_type_choice is not None
        and number_last_hour is not None
        and number_next_hour is not None
    ):
        utils.run_streamlit(
            area_choice=area_choice,
            device_type_choice=int(device_type_choice),
            number_last_hour=int(number_last_hour),
            number_next_hour=int(number_next_hour),
        )
        st.image("application/api/static/plot.png")


if __name__ == "__main__":
    main()
