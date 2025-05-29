# --- Import Libraries ---
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Crop Protection Innovation Dashboard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stSlider [data-baseweb="slider"] {
            padding: 0;
        }
        .main {
            padding: 2rem;
        }
        .sidebar .sidebar-content {
            padding: 1rem;
        }
        .stProgress > div > div > div > div {
            background-color: #2e8b57;
        }
        .st-bb {
            background-color: transparent;
        }
        .st-at {
            background-color: #2e8b57;
        }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- Data Loading Function ---
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_data():
    """Load the survey data from the embedded dataset"""
    try:
        df = pd.read_excel('survey_data.xlsx')
        
        # Store total records count before any processing
        total_records = len(df)
        
        # Replace submitdate with G01Q46 contents
        df['submitdate'] = pd.to_datetime(df['G01Q46'], errors='coerce')
        
        # Remove comma separators from seed column if they exist
        if 'seed' in df.columns:
            df['seed'] = df['seed'].astype(str).str.replace(',', '')
        
        # Identify pesticide data columns
        pest_cols = [col for col in df.columns if 'G03Q19' in col]
        df = convert_pesticide_columns(df, pest_cols)
        
        # Preprocess data
        df['G00Q01'] = df['G00Q01'].str.strip()
        df['G00Q03'] = df['G00Q03'].str.strip()
        
        return df, total_records
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, 0

# --- Data Processing Functions ---
def clean_numeric(value):
    """Convert various numeric formats to float, handling text entries"""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    
    # Remove non-numeric characters except decimal points and negative signs
    cleaned = re.sub(r"[^\d.-]", "", str(value))
    try:
        return float(cleaned) if cleaned else np.nan
    except ValueError:
        return np.nan

def convert_pesticide_columns(df, cols):
    """Convert pesticide-related columns to numeric values"""
    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric)
    return df

def process_text_columns(df, columns):
    """Process text columns for analysis, handling numpy arrays"""
    all_text = []
    for col in columns:
        if col in df.columns:
            # Convert to string and handle NaN values
            text_series = df[col].astype(str).replace('nan', '')
            all_text.extend(text_series.tolist())
    return ' '.join([str(t) for t in all_text if str(t) != ''])

def generate_wordcloud(text, title, colormap='viridis'):
    """Generate and display a word cloud"""
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap=colormap,
        max_words=100,
        contour_width=1,
        contour_color='steelblue'
    ).generate(text)
    
    # Display the generated image
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=16, pad=20)
    ax.axis('off')
    st.pyplot(fig)

# --- Visualization Functions ---
def create_bar_chart(df, x_col, y_col, title, color='steelblue'):
    """Create an Altair bar chart"""
    chart = alt.Chart(df).mark_bar(color=color).encode(
        x=alt.X(f'{x_col}:Q', title=x_col),
        y=alt.Y(f'{y_col}:N', title=y_col, sort='-x')
    ).properties(
        title=title,
        width=600,
        height=400
    )
    return chart

def create_line_chart(df, x_col, y_col, color_col, title):
    """Create an Altair line chart"""
    chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X(f'{x_col}:N', title=x_col),
        y=alt.Y(f'{y_col}:Q', title=y_col),
        color=alt.Color(f'{color_col}:N', title=color_col),
        tooltip=[x_col, y_col, color_col]
    ).properties(
        title=title,
        width=600,
        height=400
    )
    return chart

def create_word_frequency_chart(text, title):
    """Create word frequency visualization"""
    words = re.findall(r'\b\w{4,}\b', text.lower())
    word_counts = Counter(words)
    word_df = pd.DataFrame(word_counts.most_common(20), columns=['word', 'count'])
    
    chart = alt.Chart(word_df).mark_bar().encode(
        x='count:Q',
        y=alt.Y('word:N', sort='-x'),
        color=alt.Color('count:Q', scale=alt.Scale(scheme='blues'))
    ).properties(
        title=title,
        width=600,
        height=400
    )
    return chart

# --- Survey Monitoring Dashboard Functions ---
def show_kpi_cards(df, total_records):
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate valid responses (non-empty)
    valid_responses = len(df)
    invalid_responses = total_records - valid_responses
    
    # Display with red font for incomplete count
    col1.markdown(f"""
    <div style="border-radius:10px; padding:10px; background-color:#f0f2f6">
        <h3 style="margin:0; padding:0">Total Responses</h3>
        <p style="margin:0; padding:0; font-size:24px">
            {valid_responses} <span style="color:red">({invalid_responses} incomplete)</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col2.metric("Countries Represented", df['G00Q01'].nunique())
    col3.metric("Regulators", len(df[df['G00Q03'] == "Regulator"]))
    col4.metric("Industry Representatives", len(df[df['G00Q03'] == "Industry"]))

def show_response_overview(df):
    st.subheader("Response Overview")
    tab1, tab2, tab3 = st.tabs(["By Country", "By Stakeholder", "Over Time"])
    
    with tab1:
        country_counts = df['G00Q01'].value_counts().reset_index()
        country_counts.columns = ['Country', 'Count']
        chart = create_bar_chart(country_counts, 'Count', 'Country', 'Responses by Country')
        st.altair_chart(chart, use_container_width=True)
    
    with tab2:
        stakeholder_counts = df['G00Q03'].value_counts().reset_index()
        stakeholder_counts.columns = ['Stakeholder', 'Count']
        chart = create_bar_chart(stakeholder_counts, 'Count', 'Stakeholder', 'Responses by Stakeholder')
        st.altair_chart(chart, use_container_width=True)
    
    with tab3:
        time_df = df.set_index('submitdate').resample('W').size().reset_index(name='counts')
        time_df.columns = ['Date', 'Count']
        chart = alt.Chart(time_df).mark_line().encode(
            x='Date:T',
            y='Count:Q',
            tooltip=['Date', 'Count']
        ).properties(
            title='Responses Over Time',
            width=800,
            height=400
        )
        st.altair_chart(chart, use_container_width=True)

def show_policy_analysis(df):
    st.subheader("Policy and Regulation Analysis")
    
    # Policy presence
    st.markdown("**Policy and Regulatory Framework Presence**")
    policy_cols = [
        'G00Q11.SQ001_SQ001.', 'G00Q11.SQ002_SQ001.', 
        'G00Q11.SQ003_SQ001.', 'G00Q11.SQ004_SQ001.'
    ]
    policy_names = [
        "Pesticide Policy", "Conventional Pesticide Legislation",
        "Biopesticide Legislation", "IP Protection Legislation"
    ]
    
    policy_df = pd.DataFrame({
        'Policy': policy_names,
        'Yes': [df[col].str.contains('Yes').sum() for col in policy_cols],
        'No': [df[col].str.contains('No').sum() for col in policy_cols]
    }).melt(id_vars='Policy', var_name='Response', value_name='Count')
    
    chart = alt.Chart(policy_df).mark_bar().encode(
        x='Count:Q',
        y='Policy:N',
        color='Response:N',
        tooltip=['Policy', 'Response', 'Count']
    ).properties(
        width=600,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)
    
    # Innovation Ratings
    st.markdown("**Innovation Enabling Ratings (1-5 scale)**")
    rating_cols = [
        'G00Q14.SQ001.', 'G00Q14.SQ002.', 'G00Q14.SQ003.', 
        'G00Q14.SQ004.', 'G00Q14.SQ006.', 'G00Q14.SQ007.'
    ]
    rating_names = [
        "Digital Technologies", "Biotechnology", "Renewable Energy",
        "Artificial Intelligence", "Conventional Pesticides", "Biopesticides"
    ]
    
    rating_df = df[rating_cols].apply(pd.to_numeric, errors='coerce')
    rating_df = rating_df.mean().reset_index()
    rating_df.columns = ['Innovation', 'Average Rating']
    rating_df['Innovation'] = rating_names
    
    chart = alt.Chart(rating_df).mark_bar().encode(
        x='Average Rating:Q',
        y=alt.Y('Innovation:N', sort='-x'),
        color=alt.Color('Average Rating:Q', scale=alt.Scale(scheme='greens')),
        tooltip=['Innovation', 'Average Rating']
    ).properties(
        title='Average Innovation Enabling Ratings',
        width=600,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)

def show_registration_process(df):
    st.subheader("Pesticide Registration Process")
    reg_cols = [
        'G00Q18.SQ001_SQ001.', 'G00Q18.SQ002_SQ001.', 'G00Q18.SQ003_SQ001.',
        'G00Q18.SQ004_SQ001.', 'G00Q18.SQ005_SQ001.', 'G00Q18.SQ006_SQ001.'
    ]
    reg_names = [
        "Dossier Submission", "Initial Admin Actions", "Completeness Check",
        "Dossier Evaluation", "Registration Decision", "Publication"
    ]
    
    reg_df = pd.DataFrame({
        'Step': reg_names,
        'Yes': [df[col].str.contains('Yes').sum() for col in reg_cols]
    })
    
    chart = alt.Chart(reg_df).mark_bar().encode(
        x='Yes:Q',
        y=alt.Y('Step:N', sort='-x'),
        color=alt.Color('Yes:Q', scale=alt.Scale(scheme='purples')),
        tooltip=['Step', 'Yes']
    ).properties(
        title='Registration Process Steps (Conventional Pesticides)',
        width=600,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)

def show_pesticide_data(df):
    st.subheader("Pesticide Registration and Production Data")
    
    # Get all pesticide-related columns
    pest_cols = [col for col in df.columns if 'G03Q19' in col]
    
    if not df[pest_cols].empty:
        years = ['2020', '2021', '2022', '2023', '2024']
        conv_pest = []
        bio_pest = []
        
        for i in range(5):
            conv_col = f'G03Q19.SQ00{i+1}_SQ001.'
            bio_col = f'G03Q19.SQ00{i+1}_SQ002.'
            
            # Use cleaned numeric values
            conv_mean = df[conv_col].mean() if conv_col in df.columns else np.nan
            bio_mean = df[bio_col].mean() if bio_col in df.columns else np.nan
            
            conv_pest.append(conv_mean if not np.isnan(conv_mean) else 0)
            bio_pest.append(bio_mean if not np.isnan(bio_mean) else 0)
        
        pest_df = pd.DataFrame({
            'Year': years,
            'Conventional Pesticides': conv_pest,
            'Biopesticides': bio_pest
        }).melt(id_vars='Year', var_name='Type', value_name='Count')
        
        chart = create_line_chart(pest_df, 'Year', 'Count', 'Type', 
                                'Average Number of Registered Pesticides')
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("No pesticide registration data available")

def show_adoption_metrics(df):
    st.subheader("Adoption and Awareness")
    
    # Implementation of innovations
    st.markdown("**Implementation of Innovations (1-5 scale)**")
    impl_cols = [
        'G04Q21.SQ001.', 'G04Q21.SQ002.', 'G04Q21.SQ003.', 'G04Q21.SQ004.'
    ]
    impl_names = [
        "IPM Implementation", "CRISPR Gene Editing", 
        "Advanced Monitoring Systems", "Targeted Pest Behavior Studies"
    ]
    
    impl_df = df[impl_cols].apply(pd.to_numeric, errors='coerce').mean().reset_index()
    impl_df.columns = ['Innovation', 'Average Rating']
    impl_df['Innovation'] = impl_names
    
    chart = alt.Chart(impl_df).mark_bar().encode(
        x='Average Rating:Q',
        y=alt.Y('Innovation:N', sort='-x'),
        color=alt.Color('Average Rating:Q', scale=alt.Scale(scheme='oranges')),
        tooltip=['Innovation', 'Average Rating']
    ).properties(
        width=600,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)
    
    # Farmer awareness and access
    st.markdown("**Farmer Awareness and Access**")
    awareness_cols = ['G00Q30.SQ001.', 'G00Q30.SQ002.']
    awareness_df = df[awareness_cols].apply(pd.to_numeric, errors='coerce').mean().reset_index()
    awareness_df.columns = ['Metric', 'Average Rating']
    awareness_df['Metric'] = ['Awareness', 'Access']
    
    chart = alt.Chart(awareness_df).mark_bar().encode(
        x='Metric:N',
        y='Average Rating:Q',
        color=alt.Color('Average Rating:Q', scale=alt.Scale(scheme='teals')),
        tooltip=['Metric', 'Average Rating']
    ).properties(
        title='Farmer Awareness and Access (1-5 scale)',
        width=600,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)

def show_text_analysis(df, title, columns):
    """Display text analysis for challenges/recommendations with word cloud"""
    st.markdown(f"### {title}")
    
    # Process text columns safely
    text = process_text_columns(df, columns)
    
    if text.strip():
        # Create two columns: word cloud and frequency chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Word Cloud Visualization**")
            generate_wordcloud(
                text, 
                title,
                colormap='RdYlGn' if 'Challenge' in title else 'viridis'
            )
        
        with col2:
            st.markdown("**Top 20 Keywords**")
            chart = create_word_frequency_chart(text, f"Most Frequent Terms in {title}")
            st.altair_chart(chart, use_container_width=True)
        
        # Show full text in expander
        with st.expander(f"View all {title.lower()}"):
            st.text(text[:5000])  # Limit to first 5000 chars
    else:
        st.warning(f"No {title.lower()} data available")

def survey_monitoring_dashboard():
    st.title("🌾 Crop Protection Innovation Survey Dashboard")
    st.markdown("Monitoring the flow of crop protection innovation in low- and middle-income countries")
    
    # Load data
    df, total_records = load_data()
    
    if df is None:
        st.stop()
    
    # Sidebar filters
    st.sidebar.title("Filters")
    
    with st.sidebar.expander("Select Countries", expanded=False):
        selected_countries = st.multiselect(
            "Countries",
            options=df['G00Q01'].unique(),
            default=df['G00Q01'].unique(),
            label_visibility="collapsed"
        )
    
    with st.sidebar.expander("Select Stakeholder Categories", expanded=False):
        selected_stakeholders = st.multiselect(
            "Stakeholders",
            options=df['G00Q03'].dropna().unique(),
            default=df['G00Q03'].dropna().unique(),
            label_visibility="collapsed"
        )
    
    # Date range selection - clean and compact
    if not df['submitdate'].isna().all():
        min_date = df['submitdate'].min().date()
        max_date = max(df['submitdate'].max().date(), datetime.today().date())
        
        st.sidebar.markdown("**Select Date Range**")
        
        # Create a clean date range selector
        col1, col2 = st.sidebar.columns([1, 3])
        with col1:
            st.markdown("From:")
        with col2:
            start_date = st.date_input(
                "Start date",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                label_visibility="collapsed"
            )
        
        col1, col2 = st.sidebar.columns([1, 3])
        with col1:
            st.markdown("To:")
        with col2:
            end_date = st.date_input(
                "End date",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                label_visibility="collapsed"
            )
    else:
        today = datetime.today().date()
        start_date = end_date = today
    
    # Apply filters
    filtered_df = df[
        (df['G00Q01'].isin(selected_countries)) &
        (df['G00Q03'].isin(selected_stakeholders))
    ].copy()
    
    # Apply date filter if we have dates
    if not df['submitdate'].isna().all():
        filtered_df = filtered_df[
            (filtered_df['submitdate'].dt.date >= start_date) &
            (filtered_df['submitdate'].dt.date <= end_date)
        ]
    
    if filtered_df.empty:
        st.warning("No data matches the selected filters")
        return
    
    # Dashboard sections
    show_kpi_cards(filtered_df, total_records)
    show_response_overview(filtered_df)
    show_policy_analysis(filtered_df)
    show_registration_process(filtered_df)
    show_pesticide_data(filtered_df)
    show_adoption_metrics(filtered_df)
    
    # Challenges and Recommendations
    st.subheader("Text Analysis")
    show_text_analysis(filtered_df, "Common Challenges", 
                      ['G00Q36', 'G00Q37', 'G00Q38', 'G00Q39', 'G00Q40', 'G00Q41'])
    show_text_analysis(filtered_df, "Key Recommendations", 
                      ['G00Q42', 'G00Q43', 'G00Q44', 'G00Q45'])
    
    # Data explorer
    st.subheader("Data Explorer")
    if st.checkbox("Show raw data"):
        cols_to_show = [col for col in filtered_df.columns 
                       if col not in ['lastpage', 'startlanguage', 'G01Q46']]
        display_df = filtered_df[cols_to_show]
        st.dataframe(display_df)
    
    # Download button
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')
    
    csv = convert_df(filtered_df)
    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name='filtered_survey_data.csv',
        mime='text/csv'
    )
    
    # Footer
    st.markdown("---")
    st.markdown("**Crop Protection Innovation Survey Dashboard** · Powered by Virtual Analytics")

# --- Analysis Dashboard Functions ---
# --- Analysis Dashboard Functions ---
def analysis_dashboard():
    st.title("🌱 Crop Protection Innovation Dashboard")
    st.markdown("""
    This dashboard provides comprehensive analysis of crop protection innovation flow in low- and middle-income countries, 
    focusing on technology, sustainability, and productivity.
    """)

    # Load data
    df, total_records = load_data()
    
    if df is None:
        st.stop()
    
    # Sidebar filters - now collapsible by default
    st.sidebar.header("Filter Data")
    
    with st.sidebar.expander("Select Countries", expanded=False):
        selected_countries = st.multiselect(
            "Countries",
            options=df['G00Q01'].unique(),
            default=df['G00Q01'].unique(),
            label_visibility="collapsed"
        )

    with st.sidebar.expander("Select Stakeholder Types", expanded=False):
        selected_stakeholders = st.multiselect(
            "Stakeholders",
            options=df['G00Q03'].unique(),
            default=df['G00Q03'].unique(),
            label_visibility="collapsed"
        )

    # Filter data
    filtered_df = df[
        (df['G00Q01'].isin(selected_countries)) & 
        (df['G00Q03'].isin(selected_stakeholders))
    ]

    # Overview section
    st.header("📊 Overview of Survey Responses")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Responses", len(df))
    col2.metric("Countries Represented", df['G00Q01'].nunique())
    col3.metric("Stakeholder Types", df['G00Q03'].nunique())

    # Country and stakeholder distribution
    st.subheader("Geographical and Stakeholder Distribution")

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.countplot(data=filtered_df, y='G00Q01', order=filtered_df['G00Q01'].value_counts().index, ax=ax1)
    ax1.set_title('Responses by Country')
    ax1.set_xlabel('Number of Responses')
    ax1.set_ylabel('Country')
    st.pyplot(fig1)
    st.caption("""
    **Insight:** The survey responses span a wide range of countries, but participation is uneven. Zambia, Nigeria, and Ethiopia recorded the highest number of responses, suggesting greater stakeholder engagement or easier access to respondents in these countries. Kenya, Tanzania, and Angola also contributed significantly. Countries like Malawi, Saudi Arabia, and South Africa had the least representation, which may reflect either limited stakeholder engagement, outreach challenges, or a smaller crop protection innovation footprint.
    """)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.countplot(data=filtered_df, y='G00Q03', order=filtered_df['G00Q03'].value_counts().index, ax=ax2)
    ax2.set_title('Responses by Stakeholder Type')
    ax2.set_xlabel('Number of Responses')
    ax2.set_ylabel('Stakeholder Type')
    st.pyplot(fig2)
    st.caption("""
    **Insight:** The majority of responses come from industry players and regulators, reflecting their central role in crop protection innovation ecosystems. This dominance suggests that regulatory compliance and commercial product development are key drivers of innovation flow. However, the notably low participation from farmers, researchers, and academia highlights a critical gap in inclusive innovation. The underrepresentation of these groups may limit the practical relevance, field-level adoption, and research-driven refinement of crop protection technologies.
    """)

    # Policy and Regulation Analysis
    st.header("📜 Policy and Regulatory Environment")

    # Policy existence
    st.subheader("Existence of Key Policies")

    policy_cols = {
        'Pesticide Policy': 'G00Q11.SQ001_SQ001.',
        'Biosafety Policy': 'G00Q11.SQ002_SQ001.',
        'IPM Policy': 'G00Q11.SQ003_SQ001.',
        'Drone Policy': 'G00Q11.SQ004_SQ001.'
    }

    policy_data = []
    for policy_name, col_prefix in policy_cols.items():
        # Find columns that start with this prefix
        cols = [c for c in df.columns if c.startswith(col_prefix)]
        if cols:
            # Convert to string and clean
            policy_series = df[cols[0]].astype(str).str.strip().str.lower()

            # Count responses
            yes_count = policy_series.str.contains('yes', na=False).sum()
            no_count = policy_series.str.contains('no', na=False).sum()
            missing_count = policy_series.isna().sum()

            policy_data.append({
                'Policy': policy_name,
                'Yes': yes_count,
                'No': no_count,
                'Missing': missing_count
            })

    policy_df = pd.DataFrame(policy_data).set_index('Policy')

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    policy_df[['Yes', 'No']].plot(kind='barh', stacked=True, ax=ax3)
    ax3.set_title('Existence of Key Policies')
    ax3.set_xlabel('Number of Responses')
    st.pyplot(fig3)
    st.caption("""
    **Insight: Pesticide policies** are the most widely reported, indicating they are well-established and likely more mature across countries. In contrast, drone policies are the least common, underscoring regulatory lag in adapting to emerging technologies. The relatively low existence of Integrated Pest Management (IPM) policies is a notable gap—especially given the global shift toward sustainable and ecological farming practices. This suggests an opportunity for countries to scale up IPM policy frameworks to promote safer, more sustainable crop protection.
    """)

    # Regulatory effectiveness - Fixed to handle numeric conversion
    st.subheader("Perceived Effectiveness of Regulatory Processes")

    effectiveness_cols = {
        'Registration Process': 'G00Q12.SQ001_SQ001.',
        'Post-Market Surveillance': 'G00Q12.SQ001_SQ002.',
        'Data Protection': 'G00Q12.SQ001_SQ003.',
        'Enforcement': 'G00Q12.SQ001_SQ004.',
        'Label Approval': 'G00Q12.SQ002_SQ001.',
        'Import Control': 'G00Q12.SQ002_SQ002.',
        'Export Control': 'G00Q12.SQ002_SQ003.',
        'Disposal': 'G00Q12.SQ002_SQ004.'
    }

    effectiveness_data = []
    for process_name, col_prefix in effectiveness_cols.items():
        cols = [c for c in df.columns if c.startswith(col_prefix)]
        if cols:
            # Convert to numeric safely
            rating_series = pd.to_numeric(df[cols[0]], errors='coerce')
            avg_rating = rating_series.mean()
            effectiveness_data.append({
                'Process': process_name,
                'Average Rating': avg_rating
            })

    effectiveness_df = pd.DataFrame(effectiveness_data).set_index('Process')

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=effectiveness_df, y=effectiveness_df.index, x='Average Rating', ax=ax4)
    ax4.set_title('Average Rating of Regulatory Process Effectiveness (1-5 scale)')
    ax4.set_xlabel('Average Rating')
    ax4.set_xlim(0, 5)
    st.pyplot(fig4)
    st.caption("""
    **Insight:** Most regulatory processes are rated moderately effective, reflecting a system that is functional but with evident performance gaps. Data protection stands out as the most positively rated, potentially reflecting greater institutional clarity or investment in this area. In contrast, disposal and export control receive the lowest effectiveness ratings—flagging critical regulatory blind spots. These gaps likely pose environmental and trade risks, respectively, and highlight the urgent need for reforms to strengthen enforcement, safe disposal mechanisms, and streamlined export protocols for crop protection products.
    """)

    # Innovation Flow Analysis
    st.header("💡 Innovation Flow and Adoption")

    # Time for registration
    st.subheader("Time Taken for Product Registration")

    time_cols = {
        'Conventional Pesticides': 'G00Q14.SQ001.',
        'Biopesticides': 'G00Q14.SQ002.',
        'Biocontrol Agents': 'G00Q14.SQ003.',
        'New Technologies': 'G00Q14.SQ004.'
    }

    time_data = []
    for tech_name, col in time_cols.items():
        if col in df.columns:
            time_counts = df[col].value_counts().to_dict()
            for time_period, count in time_counts.items():
                time_data.append({
                    'Technology': tech_name,
                    'Time Period': time_period,
                    'Count': count
                })

    time_df = pd.DataFrame(time_data)

    fig5 = px.bar(time_df, x='Technology', y='Count', color='Time Period', 
                  title='Time Taken for Product Registration by Technology Type')
    st.plotly_chart(fig5, use_container_width=True)
    st.caption("""
    **Insight:** The data clearly shows that conventional pesticides benefit from faster and more predictable registration timelines, likely due to more established and well-understood regulatory pathways. In contrast, biopesticides, biocontrol agents, and newer technologies experience more prolonged approval times, with a notable concentration in the 3-to-5 year range. This delay suggests that regulatory systems are not yet fully adapted to accommodate emerging innovations, potentially slowing down the adoption of safer, more sustainable alternatives. Harmonizing and updating regulatory frameworks to accelerate review processes for newer technologies could unlock significant benefits in innovation uptake and sustainable agriculture practices.
    """)

    # Innovation adoption challenges
    st.subheader("Challenges in Adopting New Technologies")

    # Text analysis of challenges
    challenge_cols = {
        'General Challenges': 'G00Q39',
        'Regulatory Challenges': 'G00Q40',
        'Biopesticide Challenges': 'G00Q41',
        'Biocontrol Challenges': 'G00Q42'
    }

    # Word cloud for general challenges
    if 'G00Q39' in df.columns:
        text = ' '.join(df['G00Q39'].dropna().astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        fig6, ax6 = plt.subplots(figsize=(10, 6))
        ax6.imshow(wordcloud, interpolation='bilinear')
        ax6.axis('off')
        ax6.set_title('Word Cloud of General Innovation Adoption Challenges')
        st.pyplot(fig6)
        st.caption("""
        **Insight:** The most pressing challenges in adopting new crop protection technologies center around regulatory bottlenecks, particularly the lack of specific guidelines for biopesticides and biocontrol agents, and unclear or lengthy registration processes. Terms like “lack,” “guideline,” “review,” “efficacy,” and “regulation” dominate the word cloud, pointing to significant gaps in policy clarity and institutional readiness.
        Additionally, the frequent appearance of “farmers,” “skills,” “training,” and “illiteracy” highlights the limited farmer awareness and technical capacity, indicating that extension services and field-based education programs remain critically underfunded or underutilized.
        Financial and operational challenges are also apparent, with words like “cost,” “access,” and “resources” pointing to limited financial incentives or subsidies to support innovation adoption.
        Strategic Implication:
        Improving the adoption of innovations will require:
        •	Tailored regulatory frameworks for new technologies (e.g., separate dossiers and review protocols for biopesticides).
        •	Targeted farmer training and capacity-building initiatives.
        •	Strengthened coordination among regulators, researchers, and private sector actors to address institutional and knowledge gaps.

        """)

    # Sentiment analysis of challenges
    if 'G00Q39' in df.columns:
        sentiments = []
        for text in df['G00Q39'].dropna():
            blob = TextBlob(str(text))
            sentiments.append(blob.sentiment.polarity)
        
        fig7, ax7 = plt.subplots(figsize=(10, 6))
        sns.histplot(sentiments, bins=20, kde=True, ax=ax7)
        ax7.set_title('Sentiment Analysis of Innovation Challenge Descriptions')
        ax7.set_xlabel('Sentiment Polarity (-1 to 1)')
        ax7.set_ylabel('Frequency')
        st.pyplot(fig7)
        st.caption("""
        **Insight:** The sentiment distribution of innovation challenge descriptions is overwhelmingly neutral, with a slight skew toward mildly negative sentiment. This suggests that while stakeholders are not overly pessimistic, their language does reflect underlying concerns, frustrations, or bureaucratic fatigue in adopting new technologies. The limited presence of positive sentiment and the clustering around zero polarity indicate that stakeholders tend to describe challenges factually rather than emotionally, focusing on practical obstacles rather than voicing optimism or deep dissatisfaction.
Interpretation:
•	The sentiment landscape reflects realism rather than resistance—a sign that respondents are engaged but constrained.
•	The absence of extreme negativity may suggest constructive criticism rather than outright disapproval, presenting an opportunity to act on these insights.
Strategic Implication:
Efforts to support innovation should be framed as collaborative solutions, responding to the practical tone of feedback—through policy clarity, faster processes, and support mechanisms—rather than simply motivational or awareness-based campaigns.
        """)

    # Technology Impact Assessment
    st.header("📈 Technology Impact Assessment")

    # Technology adoption ratings
    tech_cols = {
        'Increased Productivity': 'G00Q24.SQ001.',
        'Improved Sustainability': 'G00Q24.SQ002.',
        'Enhanced Food Safety': 'G00Q24.SQ003.'
    }

    tech_data = []
    for impact_name, col in tech_cols.items():
        if col in df.columns:
            avg_rating = df[col].mean()
            tech_data.append({
                'Impact Area': impact_name,
                'Average Rating': avg_rating
            })

    tech_df = pd.DataFrame(tech_data).set_index('Impact Area')

    fig8, ax8 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=tech_df, y=tech_df.index, x='Average Rating', ax=ax8)
    ax8.set_title('Perceived Impact of Crop Protection Technologies (1-5 scale)')
    ax8.set_xlabel('Average Rating')
    ax8.set_xlim(0, 5)
    st.pyplot(fig8)
    st.caption("""
    **Insight:** Crop protection technologies are perceived to deliver the greatest benefit in increasing productivity, followed closely by enhancing food safety and then improving sustainability. This prioritization suggests that stakeholders see these technologies primarily as tools to boost agricultural output, though there is a growing recognition of their role in food system resilience and environmental stewardship.
Interpretation:
•	The high rating for productivity reflects the persistent drive to meet food demand and improve farmer yields.
•	The strong score for food safety highlights awareness of post-harvest health risks and consumer protection.
•	The slightly lower rating for sustainability implies that while important, ecological and long-term benefits may be underemphasized in policy or implementation compared to short-term gains.
Strategic Implication:
Stakeholders should consider mainstreaming sustainability metrics into technology development, promotion, and adoption strategies. Demonstrating that these innovations can simultaneously deliver yield, safety, and ecological benefits could boost acceptance and long-term impact.

    """)
    
    # Cluster analysis of respondents
    st.subheader("Stakeholder Cluster Analysis")

    # Prepare data for clustering - using only numeric columns
    cluster_cols = [
        'G00Q24.SQ001.', 'G00Q24.SQ002.', 'G00Q24.SQ003.',  # Impact ratings (should be numeric)
        'G00Q12.SQ001_SQ001.', 'G00Q12.SQ001_SQ002.',       # Regulatory effectiveness (should be numeric)
    ]

    # Check if all required columns exist and have numeric data
    available_cols = [col for col in cluster_cols if col in df.columns]
    
    # Convert all columns to numeric, coercing errors to NaN
    cluster_df = df[available_cols].apply(pd.to_numeric, errors='coerce').dropna()

    if len(cluster_df) == 0:
        st.warning("Insufficient numeric data for cluster analysis. Please check if the required columns exist and contain valid numeric data.")
    else:
        # Standardize data only if we have at least 2 samples
        if len(cluster_df) > 1:
            # Standardize data
            cluster_df_std = (cluster_df - cluster_df.mean()) / cluster_df.std()

            # Determine optimal number of clusters
            inertia = []
            max_clusters = min(6, len(cluster_df_std))  # Ensure we don't ask for more clusters than samples
            for k in range(1, max_clusters):
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(cluster_df_std)
                inertia.append(kmeans.inertia_)

            fig9, ax9 = plt.subplots(figsize=(10, 6))
            ax9.plot(range(1, max_clusters), inertia, marker='o')
            ax9.set_title('Elbow Method for Optimal Number of Clusters')
            ax9.set_xlabel('Number of Clusters')
            ax9.set_ylabel('Inertia')
            st.pyplot(fig9)
            st.caption("""
            **Insight:** The elbow plot helps determine the optimal number of clusters for segmenting stakeholders.
            """)

            # Perform clustering with 3 clusters (or fewer if not enough data)
            n_clusters = min(3, len(cluster_df_std)-1)
            if n_clusters > 0:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(cluster_df_std)

                # Add clusters to dataframe
                cluster_df['Cluster'] = cluster_labels

                # Visualize clusters with PCA if we have at least 2 dimensions
                if len(cluster_df_std.columns) >= 2:
                    pca = PCA(n_components=2)
                    pca_result = pca.fit_transform(cluster_df_std)

                    fig10, ax10 = plt.subplots(figsize=(10, 6))
                    scatter = ax10.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap='viridis')
                    ax10.set_title('PCA Visualization of Stakeholder Clusters')
                    ax10.set_xlabel('Principal Component 1')
                    ax10.set_ylabel('Principal Component 2')
                    if len(np.unique(cluster_labels)) > 1:
                        legend = ax10.legend(*scatter.legend_elements(), title="Clusters")
                        ax10.add_artist(legend)
                    st.pyplot(fig10)
                    st.caption("""
                    **Insight:** The PCA visualization shows how stakeholders group based on their perceptions.
                    """)

                # Describe clusters
                cluster_profiles = cluster_df.groupby('Cluster').mean()

                fig11, ax11 = plt.subplots(figsize=(12, 6))
                sns.heatmap(cluster_profiles.T, annot=True, cmap='YlGnBu', ax=ax11)
                ax11.set_title('Average Values by Cluster')
                st.pyplot(fig11)
                st.caption("""
                **Insight:** The heatmap reveals different stakeholder profiles based on their responses.
                """)
            else:
                st.warning("Not enough data points to perform clustering.")
        else:
            st.warning("Not enough data points to determine optimal clusters.")

    # Predictive Modeling
 
    st.header("🔮 Predictive Analysis")

    # Check if required columns exist
    required_cols = ['G00Q03', 'G00Q24.SQ001.', 'G00Q24.SQ002.', 'G00Q24.SQ003.', 
                    'G00Q12.SQ001_SQ001.', 'G00Q12.SQ001_SQ002.']
    available_cols = [col for col in required_cols if col in df.columns]

    if len(available_cols) == len(required_cols):
        # Prepare data for prediction
        model_df = df[required_cols].dropna()

        if len(model_df) > 10:  # Minimum threshold for meaningful analysis
            # Encode stakeholder type
            le = LabelEncoder()
            model_df['Stakeholder_Encoded'] = le.fit_transform(model_df['G00Q03'])
    
            # Define target (high impact on productivity)
            model_df['High_Impact'] = (model_df['G00Q24.SQ001.'] >= 4).astype(int)
    
            # Features and target
            X = model_df[['Stakeholder_Encoded', 'G00Q12.SQ001_SQ001.', 'G00Q12.SQ001_SQ002.']]
            y = model_df['High_Impact']
    
            # Adjust test size based on available data
            test_size = min(0.3, 0.9 * len(X) / len(X))  # Ensure we leave at least 10% for training
    
            # Train-test split with validation
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=test_size, 
                    random_state=42,
                    stratify=y
                )
        
                # Only proceed if we have samples in both sets
                if len(X_train) > 0 and len(X_test) > 0:
                    # Train model
                    rf = RandomForestClassifier(random_state=42)
                    rf.fit(X_train, y_train)
            
                    # Evaluate
                    y_pred = rf.predict(X_test)
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
            
                    st.subheader("Classification Report for Predicting High Productivity Impact")
                    st.dataframe(report_df.style.format("{:.2f}"))
                    st.caption("""
                    **Insight:** The model predicts whether stakeholders will rate technologies as having 
                    high productivity impact based on their type and regulatory effectiveness perceptions.
                    """)
            
                    # Feature importance
                    importance_df = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': rf.feature_importances_
                    }).sort_values('Importance', ascending=False)
            
                    fig12, ax12 = plt.subplots(figsize=(10, 6))
                    sns.barplot(data=importance_df, y='Feature', x='Importance', ax=ax12)
                    ax12.set_title('Feature Importance for Predicting High Productivity Impact')
                    st.pyplot(fig12)
                    st.caption("""
                    **Insight:** Shows which factors most influence perceptions of technology impact.
                    """)
                else:
                    st.warning("Insufficient data after train-test split to perform analysis.")
        
            except ValueError as e:
                st.warning(f"Could not perform predictive analysis: {str(e)}")
        else:
            st.warning(f"Insufficient data for predictive analysis (only {len(model_df)} valid records). Need at least 10.")
    else:
        missing_cols = set(required_cols) - set(available_cols)
        st.warning(f"Cannot perform predictive analysis. Missing required columns: {missing_cols}")

    # Prescriptive Recommendations
    st.header("💡 Prescriptive Recommendations")

    # Generate recommendations based on analysis
    recommendations = [
        {
            "Area": "Regulatory Systems",
            "Recommendation": "Strengthen post-market surveillance and enforcement mechanisms to improve confidence in crop protection technologies.",
            "Rationale": "Analysis showed these are the weakest aspects of regulatory systems but most predictive of positive technology perceptions."
        },
        {
            "Area": "Farmer Engagement",
            "Recommendation": "Increase farmer participation in innovation systems through targeted outreach and education programs.",
            "Rationale": "Farmers were underrepresented in survey responses but are critical end-users of technologies."
        },
        {
            "Area": "Technology Development",
            "Recommendation": "Prioritize development of biopesticides and biocontrol agents with streamlined regulatory pathways.",
            "Rationale": "These technologies face longer registration times despite their sustainability benefits."
        },
        {
            "Area": "Policy Framework",
            "Recommendation": "Develop specific policies for emerging technologies like drone applications in agriculture.",
            "Rationale": "Drone policies were the least commonly reported among surveyed countries."
        },
        {
            "Area": "Capacity Building",
            "Recommendation": "Invest in training for regulators on evaluating new technologies and for farmers on adopting them.",
            "Rationale": "Knowledge gaps were frequently cited as barriers to innovation adoption."
        }
    ]

    rec_df = pd.DataFrame(recommendations)

    st.table(rec_df)
    st.caption("""
    These recommendations are derived from patterns identified in the survey data analysis and aim to address 
    the key challenges and opportunities revealed through the research.
    """)

    # Country-specific insights
    st.header("🌍 Country-Specific Insights")

    if 'G00Q01' in df.columns:
        # First convert all numeric columns to numeric type
        numeric_cols = ['G00Q24.SQ001.', 'G00Q24.SQ002.', 'G00Q12.SQ001_SQ001.']
    
        # Create a copy of the dataframe with numeric conversions
        country_df = df.copy()
        for col in numeric_cols:
            if col in country_df.columns:
                country_df[col] = pd.to_numeric(country_df[col], errors='coerce')
    
        # Handle the registration time column separately
        most_common_time = None
        if 'G00Q14.SQ001.' in country_df.columns:
            # First clean the registration time strings
            time_mapping = {
                'below 1 year': '0-1 year',
                '1-2 years': '1-2 years',
                '2-3 years': '2-3 years',
                'above 3 years': '3+ years',
                'less than 1 year': '0-1 year',
                'more than 3 years': '3+ years'
            }
        
            # Clean and standardize the time strings
            country_df['G00Q14.SQ001.'] = (
                country_df['G00Q14.SQ001.']
                .astype(str)
                .str.strip()
                .str.lower()
                .replace(time_mapping)
            )
        
            # Get the most common registration time per country
            most_common_time = (
                country_df.groupby('G00Q01')['G00Q14.SQ001.']
                .apply(lambda x: x.mode()[0] if not x.mode().empty else 'Not Available')
            )
    
        # Calculate statistics for numeric columns
        stats = {}
        for col in numeric_cols:
            if col in country_df.columns:
                stats[f'avg_{col}'] = country_df.groupby('G00Q01')[col].mean()
    
        # Combine all statistics
        country_stats = pd.DataFrame(stats)
    
        # Rename columns for better display
        country_stats = country_stats.rename(columns={
            'avg_G00Q24.SQ001.': 'Avg_Productivity_Impact',
            'avg_G00Q24.SQ002.': 'Avg_Sustainability_Impact',
            'avg_G00Q12.SQ001_SQ001.': 'Avg_Registration_Effectiveness'
        })
    
        # Add the most common registration time if available
        if most_common_time is not None:
            country_stats['Most_Common_Registration_Time'] = most_common_time
    
        # Sort by productivity impact
        if not country_stats.empty:
            country_stats = country_stats.sort_values('Avg_Productivity_Impact', ascending=False)
    
        st.subheader("Country Performance Metrics")
    
        # Display the numeric columns with formatting
        if not country_stats.empty:
            # Format numeric columns
            formatted_stats = country_stats.copy()
            styled_df = country_stats.style.format({
                col: "{:.2f}" for col in country_stats.columns if col.startswith('Avg_')
            }).background_gradient(
                cmap='YlGnBu',
                subset=[col for col in country_stats.columns if col.startswith('Avg_')]
            )
            # Create styled dataframe
            styled_df = formatted_stats.style.background_gradient(
                cmap='YlGnBu',
                subset=[col for col in formatted_stats.columns if col.startswith('Avg_')]
            )
        
            # Display with improved formatting
            st.dataframe(
                styled_df,
                use_container_width=True,
                height=(min(len(formatted_stats) * 35 + 35, 500))  # Dynamic height
            )
        
            st.caption("""
            **Insight:** Countries exhibit notable disparities in how they perceive the impact of crop protection technologies and the effectiveness of related regulatory processes:
•	High Performers:
o	Mali and Saudi Arabia rate highest across all indicators — productivity, sustainability, and registration effectiveness — suggesting robust regulatory frameworks and positive technology outcomes.
o	Zimbabwe also shows strong scores in sustainability and registration despite moderate productivity.
•	Moderate Performers:
o	Kenya, Nigeria, Tanzania, Ghana, and Côte d’Ivoire demonstrate fairly balanced but mid-level performance, indicating room for growth especially in productivity or registration systems.
•	Low Performers:
o	South Africa, Zambia, Angola, and Ethiopia report low average scores, particularly in productivity and effectiveness, which may reflect bottlenecks in adoption or weak regulatory implementation.
•	Missing/Incomplete Data:
o	Uganda lacks numeric data, possibly due to limited survey input or reporting gaps, hindering its inclusion in comparative analysis.
Additional Note:
•	Countries with a lower Most Common Registration Time (e.g., Ethiopia: 1 year) may have faster but potentially less rigorous approval processes.
•	Conversely, longer registration times (e.g., Zimbabwe, Saudi Arabia: 5 years) could signal complex regulatory environments that may delay innovation unless streamlined.


            """)
        else:
            st.warning("No numeric data available for country performance metrics.")
    else:
        st.warning("Country data not available in the dataset.")

    # Download button for processed data
    st.sidebar.header("Data Export")
    if st.sidebar.button("Download Processed Data as CSV"):
        csv = filtered_df.to_csv(index=False)
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name="crop_protection_innovation_processed.csv",
            mime="text/csv"
        )

    # Footer
    st.markdown("---")
    st.markdown("""
    **Methodology Note:** 
    - Data was collected through a survey of stakeholders in low- and middle-income countries.
    - Analysis includes descriptive statistics, text mining, clustering, and predictive modeling.
    - Missing data was handled through exclusion for relevant analyses.
    """)

# --- Landing Page ---
def landing_page():
    st.title("🌍 Crop Protection Innovation Survey")
    st.markdown("""
    ## Assessing the Flow of Crop Protection Innovation in Low- and Middle-Income Countries
    
    **Subject:** Technology, Sustainability, and Productivity in Crop Protection
    
    **Objective:** This survey aims to monitor and analyze the current state of crop protection innovation 
    in low- and middle-income countries, focusing on the regulatory environment, technology adoption, 
    and barriers to innovation flow.
    """)
    
    st.image("https://images.unsplash.com/photo-1605000797499-95a51c5269ae?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", 
             use_container_width=True)
    
    st.markdown("""
    ### Select Dashboard:
    """)
    
    dashboard = st.selectbox(
        "Choose Dashboard",
        ["Survey Monitoring Dashboard", "Analysis Dashboard"],
        label_visibility="collapsed"
    )
    
    if dashboard == "Survey Monitoring Dashboard":
        survey_monitoring_dashboard()
    else:
        analysis_dashboard()

# --- Main App ---
def main():
    landing_page()

if __name__ == "__main__":
    main()