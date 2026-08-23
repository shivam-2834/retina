def load_css():
    return """
    <style>

    /* App background */
    .stApp {
        background-color: #f4f8fb;
        color: #1f2d3d;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0b3c5d;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* HEADINGS FIX (IMPORTANT) */
    h1, h2, h3 {
        color: #0b3c5d !important;
        font-weight: 700;
    }

    /* Paragraph text */
    p, span {
        color: #1f2d3d !important;
    }

    /* Inputs */
    .stTextInput input {
        background-color: #ffffff;
        color: black;
        border: 1px solid #ccc;
        border-radius: 8px;
        padding: 10px;
    }

    /* Buttons */
    .stButton>button {
        background-color: #0b3c5d;
        color: white;
        border-radius: 8px;
        padding: 10px 16px;
        border: none;
        font-weight: 600;
    }

    .stButton>button:hover {
        background-color: #145374;
    }

    /* Info box */
    .stAlert {
        border-radius: 10px;
    }

    /* Fix faded labels */
    label {
        color: #1f2d3d !important;
        font-weight: 500;
    }

    </style>
    """