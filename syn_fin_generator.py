import time
import streamlit as st
import numpy as np
import pandas as pd
from numpy.linalg import inv
import random
from numpy.core.multiarray import concatenate
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import altair as alt

# from model_module import VAE, CVAE, Encoder, Decoder
from model_module import VAE, CVAE, Encoder, Decoder

# Configuration
enc_out_dim = 100
latent_dim = 16
feat_dim = 8
beta = 3.0
max_epochs = 350
seq_n = 4
batch_size = 8
learning_rate = 1e-4
min_std = 0.25

# ==============================================================================================================================
#Load the models
sector_list = ['Consumer, Non-cyclical', 'Financial', 'Communications', 
                'Technology', 'Industrial', 'Energy', 
                'Basic Materials', 'Consumer, Cyclical', 'Utilities']

sector_abb_map = {'Basic Materials': 'BM', 
                  'Communications': 'COMM',
                  'Consumer, Cyclical': 'CSMC',
                  'Consumer, Non-cyclical': 'CSMNC',
                  'Energy': 'ENGY',
                  'Financial': 'FIN',
                  'Industrial': 'IND',
                  'Technology': 'TECH',
                  'Utilities': 'UTIL'}

vae_CSMNC = VAE(seq_n, feat_dim, enc_out_dim, latent_dim, beta, learning_rate, min_std)
cvae_CSMNC = CVAE(seq_n, feat_dim, enc_out_dim, latent_dim, beta, learning_rate, min_std)
vae_CSMNC.load_state_dict(torch.load("vae_CSMNC_checkpoint.pth"))
vae_CSMNC.eval()
cvae_CSMNC.load_state_dict(torch.load("cvae_CSMNC_checkpoint.pth"))
cvae_CSMNC.eval()

vae_BM = VAE(seq_n, feat_dim, enc_out_dim, latent_dim, beta, learning_rate, min_std)
cvae_BM = CVAE(seq_n, feat_dim, enc_out_dim, latent_dim, beta, learning_rate, min_std)
vae_BM.load_state_dict(torch.load("vae_BM_checkpoint.pth"))
vae_BM.eval()
cvae_BM.load_state_dict(torch.load("cvae_BM_checkpoint.pth"))
cvae_BM.eval()

vae_FIN = VAE(seq_n, feat_dim, enc_out_dim, latent_dim, beta, learning_rate, min_std)
cvae_FIN = CVAE(seq_n, feat_dim, enc_out_dim, latent_dim, beta, learning_rate, min_std)
vae_FIN.load_state_dict(torch.load("vae_FIN_checkpoint.pth"))
vae_FIN.eval()
cvae_FIN.load_state_dict(torch.load("cvae_FIN_checkpoint.pth"))
cvae_FIN.eval()

vae_COMM = VAE(seq_n, feat_dim, enc_out_dim, latent_dim, beta, learning_rate, min_std)
cvae_COMM = CVAE(seq_n, feat_dim, enc_out_dim, latent_dim, beta, learning_rate, min_std)
vae_COMM.load_state_dict(torch.load("vae_COMM_checkpoint.pth"))
vae_COMM.eval()
cvae_COMM.load_state_dict(torch.load("cvae_COMM_checkpoint.pth"))
cvae_COMM.eval()

vae_TECH = VAE(seq_n, feat_dim, enc_out_dim, latent_dim, beta, learning_rate, min_std)
cvae_TECH = CVAE(seq_n, feat_dim, enc_out_dim, latent_dim, beta, learning_rate, min_std)
vae_TECH.load_state_dict(torch.load("vae_TECH_checkpoint.pth"))
vae_TECH.eval()
cvae_TECH.load_state_dict(torch.load("cvae_TECH_checkpoint.pth"))
cvae_TECH.eval()

vae_IND = VAE(seq_n, feat_dim, enc_out_dim, latent_dim, beta, learning_rate, min_std)
cvae_IND = CVAE(seq_n, feat_dim, enc_out_dim, latent_dim, beta, learning_rate, min_std)
vae_IND.load_state_dict(torch.load("vae_IND_checkpoint.pth"))
vae_IND.eval()
cvae_IND.load_state_dict(torch.load("cvae_IND_checkpoint.pth"))
cvae_IND.eval()

vae_ENGY = VAE(seq_n, feat_dim, enc_out_dim, latent_dim, beta, learning_rate, min_std)
cvae_ENGY = CVAE(seq_n, feat_dim, enc_out_dim, latent_dim, beta, learning_rate, min_std)
vae_ENGY.load_state_dict(torch.load("vae_ENGY_checkpoint.pth"))
vae_ENGY.eval()
cvae_ENGY.load_state_dict(torch.load("cvae_ENGY_checkpoint.pth"))
cvae_ENGY.eval()

vae_CSMC = VAE(seq_n, feat_dim, enc_out_dim, latent_dim, beta, learning_rate, min_std)
cvae_CSMC = CVAE(seq_n, feat_dim, enc_out_dim, latent_dim, beta, learning_rate, min_std)
vae_CSMC.load_state_dict(torch.load("vae_CSMC_checkpoint.pth"))
vae_CSMC.eval()
cvae_CSMC.load_state_dict(torch.load("cvae_CSMC_checkpoint.pth"))
cvae_CSMC.eval()

vae_UTIL = VAE(seq_n, feat_dim, enc_out_dim, latent_dim, beta, learning_rate, min_std)
cvae_UTIL = CVAE(seq_n, feat_dim, enc_out_dim, latent_dim, beta, learning_rate, min_std)
vae_UTIL.load_state_dict(torch.load("vae_UTIL_checkpoint.pth"))
vae_UTIL.eval()
cvae_UTIL.load_state_dict(torch.load("cvae_UTIL_checkpoint.pth"))
cvae_UTIL.eval()

vae_dict = {'Basic Materials': vae_BM, 
            'Communications': vae_COMM,
            'Consumer, Cyclical': vae_CSMC,
            'Consumer, Non-cyclical': vae_CSMNC,
            'Energy': vae_ENGY,
            'Financial': vae_FIN,
            'Industrial': vae_IND,
            'Technology': vae_TECH,
            'Utilities': vae_UTIL
            }

cvae_dict = {'Basic Materials': cvae_BM, 
            'Communications': cvae_COMM,
            'Consumer, Cyclical': cvae_CSMC,
            'Consumer, Non-cyclical': cvae_CSMNC,
            'Energy': cvae_ENGY,
            'Financial': cvae_FIN,
            'Industrial': cvae_IND,
            'Technology': cvae_TECH,
            'Utilities': cvae_UTIL
            }

ret_rank_dict = {'Outperforming': 0,
                 'Neutral': 1,
                 'Underperforming': 2}

columns_list = [
        'Total Debts to Total Capital', 
        'Price to Book Ratio', 
        'Price Earnings Ratio (P/E)', 
        'Total Assets Growth Rate', 
        'Revenue Growth Rate', 
        'Return on Common Equity', 
        'Return on Assets', 
        'Gross Margin']
    
index_list = ['Q1', 'Q2', 'Q3', 'Q4']
log_fields = ['Total Debts to Total Capital', 'Price to Book Ratio', 'Price Earnings Ratio (P/E)']

# ==============================================================================================================================
#Functions
# Generate DF
def log_revert_field(column):
    return np.exp(column)

def generate_sample(sector, return_rank, min_std=min_std, num_preds=1):
    
    if return_rank == 'Unspecified':
        vae = vae_dict[sector]
        p = torch.distributions.Normal(torch.zeros(1, latent_dim), torch.ones(1, latent_dim))
        z = p.rsample((num_preds,)).flatten(1)
        with torch.no_grad():
            pred_mean, pred_log_scale = vae.decoder(z)
        pred_scale = torch.exp(pred_log_scale) + min_std
        e = torch.randn(num_preds, 1, 1)
        pred = pred_mean + e * pred_scale
        
    else:
        ret_rank_cat = ret_rank_dict[return_rank]
        cvae = cvae_dict[sector]
        p = torch.distributions.Normal(torch.zeros(1, latent_dim), torch.ones(1, latent_dim))
        z = p.rsample((num_preds,)).flatten(1)
        y = np.full((num_preds, 1), ret_rank_cat)
        y = torch.tensor(y, dtype=torch.float32)
        z_cond = torch.cat((z, y), dim=1)
        with torch.no_grad():
            pred_mean, pred_log_scale = cvae.decoder(z_cond)
        pred_scale = torch.exp(pred_log_scale) + min_std
        e = torch.randn(num_preds, 1, 1)
        pred = pred_mean + e * pred_scale
        
    return pred

def generate_df(year, sector, return_rank):
    
    name = sector_abb_map[sector]
    df_means, df_stds = pd.read_csv(f"df_{name}_original_means.csv"), pd.read_csv(f"df_{name}_original_std.csv")
        
    means = df_means[(year-2018)*4: (year-2018)*4+4].values
    stds = df_stds[(year-2018)*4: (year-2018)*4+4].values

    pred = generate_sample(sector=sector, return_rank=return_rank, min_std=min_std, num_preds=1)
    sample = pred.numpy()[0] * stds + means
    df_sample = pd.DataFrame(sample, columns=columns_list, index=index_list)
    for field_column in log_fields:
        df_sample[field_column] = df_sample[field_column].transform(log_revert_field)
    df_sample[['Total Assets Growth Rate', 'Revenue Growth Rate']] = df_sample[['Total Assets Growth Rate', 'Revenue Growth Rate']] * 100
    return df_sample

def generate_df_mul(year, sector, return_rank, num_pred):
    df_sample_mul = pd.DataFrame()
    
    name = sector_abb_map[sector]
    df_means = pd.read_csv(f"df_{name}_original_means.csv")
    df_stds = pd.read_csv(f"df_{name}_original_std.csv")
    
    start_index = (year - 2018) * 4
    end_index = start_index + 4

    means = df_means.iloc[start_index:end_index].values
    stds = df_stds.iloc[start_index:end_index].values

    for i in range(num_pred):  # Corrected missing colon
        ticker = f"Synthetic Company {i+1}"
        pred = generate_sample(sector=sector, return_rank=return_rank, min_std=min_std, num_preds=1)
        sample = pred.numpy()[0] * stds + means
        df_sample = pd.DataFrame(sample, columns=columns_list)
        df_sample.insert(0, 'Ticker', ticker)
        df_sample.insert(1, 'Quarter', index_list)
        df_sample_mul = pd.concat([df_sample_mul, df_sample], ignore_index=True)
        
    for field_column in log_fields:
        df_sample_mul[field_column] = df_sample_mul[field_column].transform(log_revert_field)
    
    df_sample_mul[['Total Assets Growth Rate', 'Revenue Growth Rate']] = df_sample_mul[['Total Assets Growth Rate', 'Revenue Growth Rate']] * 100
    
    return df_sample_mul

# ==============================================================================================================================
#Sidebar to the app

st.sidebar.header("**Welcome to the Synthetic Financial Data Generator!** 👋")
st.sidebar.markdown("Our generative model is built upon Variational RNN Auto-Encoders. It surpasses the performance of both the traditional Multivariate Normal Monte Carlo Model and the Multivariate Gaussian State Space Model.")


#Title and subtitle
st.title("📊 Synthetic Financial Data Generator")
st.markdown("Choose a sector and a year. This tool will then generate quarterly financial data of a made-up company in the chosen sector for the chosen year.")

#Create three columns/filters
col1, col2, col3 = st.columns(3)

with col1:
    sector = st.selectbox("Sector", sector_list, index=0)

with col2:
    year = st.selectbox("Year", ['2018', '2019', '2020', '2021', '2022'], index=0)
    year = int(year)
    
with col3:
    option = st.selectbox("Return-Tier", ['Unspecified', 'Outperforming', 'Neutral', 'Underperforming'], index=0)


#Generate One Sample
def format_percentage(value):
    return "{:.2f}%".format(value)

if st.button("Generate Data"):
    # Call the function when the button is clicked
    generated_df = generate_df(year=year, sector=sector, return_rank=option)
    
    # Format the DataFrame values
    formatted_df = generated_df.transpose().applymap(format_percentage)
    
    # Display the formatted DataFrame
    st.table(formatted_df)
    
    # Convert the formatted DataFrame back to numeric values for plotting
    numeric_df = generated_df.reset_index()
    
    # Melt the data to make it suitable for visualization
    melted_df = numeric_df.melt(id_vars='index', var_name='Field', value_name='Percentage (%)')

    # Create the Altair Chart
    chart = alt.Chart(melted_df).mark_bar(size=12.5).encode(
        x=alt.X('Field:O', axis=alt.Axis(labels=False)),  # Hide x-axis labels
        y=alt.Y('Percentage (%):Q'),
        color=alt.Color('Field:N'),
        column=alt.Column('index:N', title='Quarter'),
    ).properties(width=alt.Step(15))  # Adjust the width of the chart

    # Display Altair chart using st.altair_chart (faceted)
    # st.altair_chart(chart, use_container_width=False)
    st.bar_chart(data=generated_df, height=500)
    

#Generate Multiple Samples as Dataset
st.markdown("To generate dataset of multiple synthetic samples, select the number of samples and download the csv file.")
num_pred = st.slider('Number of samples to generate', 10, 1000, 500, 10)

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

if st.button("Generate Dataset"):
    with st.spinner(text='In progress'):
        generated_df_mul = generate_df_mul(year=year, sector=sector, return_rank=option, num_pred=num_pred)
        time.sleep(3)
    csv = convert_df(generated_df_mul)
    st.download_button(
        label="Download Dataset as CSV",
        data=csv,
        file_name='generated_data.csv',
        mime='text/csv',
    )
