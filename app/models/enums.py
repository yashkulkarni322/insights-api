from enum import Enum

class CaseType(str, Enum):
    DRUG_TRAFFICKING = "Drug Trafficking and Substance Abuse"
    ARMS_TRAFFICKING = "Arms Trafficking"
    CYBER_CRIME = "Cyber Crime"
    TERRORISM = "Terrorism"
    MURDER = "Murder and Homicide"
    SUICIDE = "Suicide"
    GENERAL = "General"

class DataSource(str, Enum):
    AUDIO = "audio"
    VIDEO = "video"
    IMAGE = "image"
    UFED = "ufed_extraction"
    OTHERS = "others"