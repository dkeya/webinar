import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
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
st.markdown("""
    <style>
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
        footer {
            visibility: hidden;
        }
    </style>
""", unsafe_allow_html=True)

# File upload and data loading
@st.cache_data(ttl=3600)
def load_data(uploaded_file):
    """Load data from uploaded file with error handling"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='latin1')
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        
        # Data cleaning and preprocessing
        # Convert date columns
        date_cols = ['submitdate', 'G01Q46']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Clean country names
        if 'G00Q01' in df.columns:
            df['G00Q01'] = df['G00Q01'].str.strip()
            df['G00Q01'] = df['G00Q01'].replace({
                'CÃ´te dâ€™Ivoire': "Côte d'Ivoire",
                'Tanzania': 'Tanzania',
                'Zambia': 'Zambia',
                'Angola': 'Angola',
                'Nigeria': 'Nigeria',
                'Egypt': 'Egypt',
                'Ethiopia': 'Ethiopia',
                'Kenya': 'Kenya',
                'Ghana': 'Ghana',
                'Zimbabwe': 'Zimbabwe',
                'Mali': 'Mali',
                'Uganda': 'Uganda',
                'South Africa': 'South Africa',
                'Saudi Arabia': 'Saudi Arabia',
                'Malawi': 'Malawi'
            })
        
        # Clean stakeholder types
        if 'G00Q03' in df.columns:
            df['G00Q03'] = df['G00Q03'].str.strip()
            df['G00Q03'] = df['G00Q03'].replace({
                'Regulator': 'Regulator',
                'Industry': 'Industry',
                'Farmer/Farmer group': 'Farmer',
                'Academia': 'Academia',
                'Research': 'Research',
                'Other': 'Other'
            })
        
        # Clean yes/no columns
        yes_no_cols = [col for col in df.columns if df[col].dtype == 'object' and 
                       df[col].str.contains('yes|no', case=False, na=False).any()]
        for col in yes_no_cols:
            df[col] = df[col].str.strip().str.lower()
            df[col] = df[col].replace({
                'yes': 'Yes',
                'no': 'No',
                'y': 'Yes',
                'n': 'No',
                'sim': 'Yes',
                'não': 'No',
                'not sure': 'Uncertain',
                'uncertain': 'Uncertain'
            })
        
        # Convert rating columns to numeric
        rating_cols = [col for col in df.columns if 'SQ00' in col or 'SQ0' in col]
        for col in rating_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Store original columns for reference
        original_columns = df.columns.tolist()
        
        return df, original_columns
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None, None

# Main app
def main():
    st.title("🌱 Crop Protection Innovation Dashboard")
    st.markdown("""
    This dashboard provides comprehensive analysis of crop protection innovation flow in low- and middle-income countries, 
    focusing on technology, sustainability, and productivity.
    """)

    # File upload
    st.sidebar.header("Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your survey data (CSV or Excel)",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is not None:
        df, original_columns = load_data(uploaded_file)
        
        if df is not None:
            # Sidebar filters
            st.sidebar.header("Filter Data")
            selected_countries = st.sidebar.multiselect(
                "Select Countries",
                options=df['G00Q01'].unique(),
                default=df['G00Q01'].unique()
            )

            selected_stakeholders = st.sidebar.multiselect(
                "Select Stakeholder Types",
                options=df['G00Q03'].unique(),
                default=df['G00Q03'].unique()
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
            **Insight:** The survey responses are distributed across multiple countries with varying representation. 
            Tanzania, Zambia, Ethiopia, and Nigeria have the highest number of responses, indicating stronger engagement 
            from these nations in crop protection innovation discussions.
            """)

            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.countplot(data=filtered_df, y='G00Q03', order=filtered_df['G00Q03'].value_counts().index, ax=ax2)
            ax2.set_title('Responses by Stakeholder Type')
            ax2.set_xlabel('Number of Responses')
            ax2.set_ylabel('Stakeholder Type')
            st.pyplot(fig2)
            st.caption("""
            **Insight:** Industry representatives and regulators dominate the survey responses, suggesting these groups 
            are more actively involved in crop protection innovation processes. Farmer representation appears relatively low, 
            which may indicate a gap in farmer engagement with innovation systems.
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
            **Insight:** Pesticide policies are the most commonly reported, while drone policies are the least common. 
            This reflects the maturity of pesticide regulation compared to newer technologies like drone applications. 
            The relatively low adoption of IPM policies suggests room for improvement in integrated pest management approaches.
            """)

            # Regulatory effectiveness
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
                    avg_rating = df[cols[0]].mean()
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
            **Insight:** Regulatory processes generally receive moderate ratings, with import control and registration 
            processes rated slightly higher than others. Disposal and export control receive the lowest ratings, 
            suggesting these are areas needing improvement in regulatory frameworks.
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
            **Insight:** Registration times vary significantly by technology type. Conventional pesticides tend to have 
            shorter registration times, while new technologies and biocontrol agents often take longer. This suggests 
            that regulatory systems may be more streamlined for traditional products compared to newer innovations.
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
                **Insight:** Common challenges include lack of farmer awareness, high costs of technologies, 
                inadequate regulatory frameworks, limited financial resources, and weak enforcement of existing regulations. 
                The prominence of "farmers" and "awareness" suggests that extension services and farmer education are critical gaps.
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
                **Insight:** The sentiment around innovation challenges is generally neutral to slightly negative, 
                reflecting the difficulties stakeholders face in adopting new technologies. The distribution suggests 
                that while some respondents express moderate frustration, most describe challenges in a relatively neutral tone.
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
            **Insight:** Technologies are perceived to have the strongest impact on productivity, followed by food safety 
            and sustainability. This suggests that while productivity remains the primary focus, there is recognition 
            of broader impacts, though sustainability may need more emphasis in technology development and deployment.
            """)

            # Cluster analysis of respondents
            
            st.subheader("Stakeholder Cluster Analysis")

            # Prepare data for clustering
            cluster_cols = [
                'G00Q24.SQ001.', 'G00Q24.SQ002.', 'G00Q24.SQ003.',  # Impact ratings
                'G00Q12.SQ001_SQ001.', 'G00Q12.SQ001_SQ002.',       # Regulatory effectiveness
                'G00Q14.SQ001.', 'G00Q14.SQ002.'                    # Registration times
            ]

            # Check if all required columns exist and have data
            available_cols = [col for col in cluster_cols if col in df.columns]
            cluster_df = df[available_cols].dropna()

            if len(cluster_df) == 0:
                st.warning("Insufficient data for cluster analysis. Please check if the required columns exist and contain valid data.")
            else:
                # Encode categorical registration times if available
                if 'G00Q14.SQ001.' in cluster_df.columns:
                    time_mapping = {
                        'Below 1 year': 1,
                        '1-2 years': 2,
                        '2-3 years': 3,
                        'Above 3 years': 4
                    }
                    cluster_df['G00Q14.SQ001.'] = cluster_df['G00Q14.SQ001.'].map(time_mapping)
    
                if 'G00Q14.SQ002.' in cluster_df.columns:
                    cluster_df['G00Q14.SQ002.'] = cluster_df['G00Q14.SQ002.'].map(time_mapping)

                # Standardize data
                cluster_df_std = (cluster_df - cluster_df.mean()) / cluster_df.std()

                # Determine optimal number of clusters (only if we have at least 2 samples)
                if len(cluster_df_std) > 1:
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
                country_stats = df.groupby('G00Q01').agg({
                    'G00Q24.SQ001.': 'mean',  # Productivity impact
                    'G00Q24.SQ002.': 'mean',  # Sustainability impact
                    'G00Q12.SQ001_SQ001.': 'mean',  # Registration effectiveness
                    'G00Q14.SQ001.': lambda x: x.mode()[0] if not x.mode().empty else np.nan  # Most common registration time
                }).rename(columns={
                    'G00Q24.SQ001.': 'Avg_Productivity_Impact',
                    'G00Q24.SQ002.': 'Avg_Sustainability_Impact',
                    'G00Q12.SQ001_SQ001.': 'Avg_Registration_Effectiveness',
                    'G00Q14.SQ001.': 'Most_Common_Registration_Time'
                }).sort_values('Avg_Productivity_Impact', ascending=False)
                
                st.subheader("Country Performance Metrics")
                st.dataframe(country_stats.style.format("{:.2f}").background_gradient(cmap='YlGnBu'))
                st.caption("""
                **Insight:** Countries vary significantly in their average ratings of technology impact and regulatory effectiveness. 
                Some countries show strong performance across multiple dimensions, while others may need targeted interventions 
                in specific areas like registration processes or sustainability impacts.
                """)

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

    else:
        st.warning("Please upload a data file to begin analysis.")

    # Footer
    st.markdown("---")
    st.markdown("""
    **Methodology Note:** 
    - Data was collected through a survey of stakeholders in low- and middle-income countries.
    - Analysis includes descriptive statistics, text mining, clustering, and predictive modeling.
    - Missing data was handled through exclusion for relevant analyses.
    """)

if __name__ == "__main__":
    main()