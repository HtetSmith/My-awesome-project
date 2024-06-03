import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Set the page configuration
st.set_page_config(
    page_title="Game Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for background and text
st.markdown(
    """
    <style>
    .main {
        background-color: black;
        background-image: url("https://www.link_to_your_background_image.com");
        background-size: cover;
    }
    h1 {
        color: #fff;
        text-shadow: 2px 2px 4px #000000;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        border: none;
        border-radius: 4px;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
    }
    .stSelectbox label, .stSlider label {
        color: #ffffff;
        text-shadow: 1px 1px 2px #000000;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Load your merged data and similarity matrix
final_merged_df = pickle.load(open('pickle/final_merged_df.pkl', 'rb'))
cosine_sim = pickle.load(open('pickle/cosine_sim.pkl', 'rb'))

# Streamlit UI components
st.title('Game Recommendation System')

# Parental control settings
required_age = st.slider('Select maximum allowed age rating:', 0, 21, 18)

# User input for game title with partial matching
selected_game_input = st.text_input('Type part or full of your favourite game title that you played:')
matching_games = final_merged_df[final_merged_df['name'].str.contains(selected_game_input, case=False, na=False)]['name'].unique()
selected_game = st.selectbox('Select a game:', matching_games)

# User input for platform, genres, and steamspy_tags
platform_modes = ['Any', 'windows', 'mac', 'linux']
selected_platform = st.selectbox('Select a platform:', platform_modes)
selected_genre = st.text_input('Enter a genre (Optional):')

# Multiplayer or single player option in selectbox with partial matching
game_modes = ['Any', 'Single-player', 'Multi-player', 'Co-op']
selected_mode = st.selectbox('Select game mode:', game_modes)

# Filter based on user inputs
filtered_df = final_merged_df.copy()

if selected_platform != 'Any':
    filtered_df = filtered_df[filtered_df['platforms'].str.contains(selected_platform, case=False, na=False)]

if selected_genre:
    filtered_df = filtered_df[filtered_df['genres'].str.contains(selected_genre, case=False, na=False)]

if selected_mode != 'Any':
    filtered_df = filtered_df[filtered_df['categories'].str.contains(selected_mode, case=False, na=False)]

# Apply required age and price filters
filtered_df = filtered_df[(filtered_df['required_age'] <= required_age) 
                          ]

# # Check the size of filtered data
# st.write(f"Filtered dataset size: {filtered_df.shape}")

# Combine relevant text data for filtered_df
filtered_df['content'] = (
    filtered_df['genres'] + ' ' + 
    filtered_df['developer'] + ' ' + 
    filtered_df['steamspy_tags'] + ' ' + 
    filtered_df['short_description']
)

# Function to get the closest 20 games based on content similarity
def get_closest_games(df, cosine_sim, top_n=20):
    if selected_game not in df['name'].values:
        st.write("The selected game is not in the filtered dataset.")
        return pd.DataFrame()  # Return an empty DataFrame

    idx = df[df['name'] == selected_game].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[:top_n]  # Top N similar games
    game_indices = [i[0] for i in sim_scores if i[0] < len(df)]
    
    return df.iloc[game_indices]

# Get closest 20 games
if selected_game:
    closest_games = get_closest_games(filtered_df, cosine_sim)
    if not closest_games.empty:
        st.write("Top 20 closest games based on your selection:")

        for index, row in closest_games.iterrows():
            st.markdown(f"### {row['name']}")
            st.image(row['header_image'])
            st.write(f"**Description:** {row['short_description']}")
            st.write(f"**Price:** ${row['price']}")
            st.write(f"**Genres:** {row['genres']}")
            st.write(f"**Developer:** {row['developer']}")
            st.write(f"**Required Age:** {row['required_age']}+")
            st.write(f"**Categories:** {row['categories']}")
            st.write("**Screenshots:**")
            
            # Display screenshots in a 3-column layout
            screenshots = eval(row['screenshots'])
            cols = st.columns(3)
            for i, screenshot in enumerate(screenshots):
                with cols[i % 3]:
                    st.image(screenshot['path_thumbnail'], use_column_width=True)
            
            st.markdown("---")
    else:
        st.write("No games found based on the current filters.")
