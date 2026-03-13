import pandas as pd
import pooch
import seaborn as sns

sns.set_theme()


def fetch_agap_gravity():
    path = pooch.retrieve(
        url="https://ramadda.data.bas.ac.uk/repository/entry/get/AGAP_BAS_Grav.XYZ?entryid=synth%3A8e5f910b-11d6-4a9d-bdf7-175c9b98cfb8%3AL0FHQVBfQkFTX0dyYXYuWFla",
        fname="AGAP_BAS_Grav.XYZ",
        path=f"{pooch.os_cache('airbornegeo')}",
        known_hash="391225810f1d15be21b37f506c098960d92af9b3ec0b48bb55dfa20e7b4cf25e",
        progressbar=True,
    )

    return pd.read_csv(path)
