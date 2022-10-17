import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import warnings
warnings.filterwarnings("ignore")



# loading in the model to predict on the data
pickle_in = open('dtc_classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

pickle_le = open('le_encoder.pkl', 'rb')
le_encoder = pickle.load(pickle_le)


# Initialize inputs disctionary with default values
input= {'age':[18],
'session_count':[1],
'gender_-unknown-':[0],
'gender_FEMALE':[0],
'gender_MALE':[0],
'gender_OTHER':[0],
'signup_method_basic':[0],
'signup_method_facebook':[0],
'signup_method_google':[0],
'signup_method_weibo':[0],
'signup_flow_0':[0],
'signup_flow_1':[0],
'signup_flow_2':[0],
'signup_flow_3':[0],
'signup_flow_4':[0],
'signup_flow_5':[0],
'signup_flow_6':[0],
'signup_flow_8':[0],
'signup_flow_10':[0],
'signup_flow_12':[0],
'signup_flow_14':[0],
'signup_flow_15':[0],
'signup_flow_16':[0],
'signup_flow_20':[0],
'signup_flow_21':[0],
'signup_flow_23':[0],
'signup_flow_24':[0],
'signup_flow_25':[0],
'language_-unknown-':[0],
'language_ca':[0],
'language_cs':[0],
'language_da':[0],
'language_de':[0],
'language_el':[0],
'language_en':[0],
'language_es':[0],
'language_fi':[0],
'language_fr':[0],
'language_hr':[0],
'language_hu':[0],
'language_id':[0],
'language_is':[0],
'language_it':[0],
'language_ja':[0],
'language_ko':[0],
'language_nl':[0],
'language_no':[0],
'language_pl':[0],
'language_pt':[0],
'language_ru':[0],
'language_sv':[0],
'language_th':[0],
'language_tr':[0],
'language_zh':[0],
'affiliate_channel_api':[0],
'affiliate_channel_content':[0],
'affiliate_channel_direct':[0],
'affiliate_channel_other':[0],
'affiliate_channel_remarketing':[0],
'affiliate_channel_sem-brand':[0],
'affiliate_channel_sem-non-brand':[0],
'affiliate_channel_seo':[0],
'affiliate_provider_baidu':[0],
'affiliate_provider_bing':[0],
'affiliate_provider_craigslist':[0],
'affiliate_provider_daum':[0],
'affiliate_provider_direct':[0],
'affiliate_provider_email-marketing':[0],
'affiliate_provider_facebook':[0],
'affiliate_provider_facebook-open-graph':[0],
'affiliate_provider_google':[0],
'affiliate_provider_gsp':[0],
'affiliate_provider_meetup':[0],
'affiliate_provider_naver':[0],
'affiliate_provider_other':[0],
'affiliate_provider_padmapper':[0],
'affiliate_provider_vast':[0],
'affiliate_provider_wayn':[0],
'affiliate_provider_yahoo':[0],
'affiliate_provider_yandex':[0],
'first_affiliate_tracked_linked':[0],
'first_affiliate_tracked_local ops':[0],
'first_affiliate_tracked_marketing':[0],
'first_affiliate_tracked_omg':[0],
'first_affiliate_tracked_product':[0],
'first_affiliate_tracked_tracked-other':[0],
'first_affiliate_tracked_untracked':[0],
'signup_app_Android':[0],
'signup_app_Moweb':[0],
'signup_app_Web':[0],
'signup_app_iOS':[0],
'first_device_type_Android Phone':[0],
'first_device_type_Android Tablet':[0],
'first_device_type_Desktop (Other)':[0],
'first_device_type_Mac Desktop':[0],
'first_device_type_Other/Unknown':[0],
'first_device_type_SmartPhone (Other)':[0],
'first_device_type_Windows Desktop':[0],
'first_device_type_iPad':[0],
'first_device_type_iPhone':[0],
'first_browser_-unknown-':[0],
'first_browser_AOL Explorer':[0],
'first_browser_Android Browser':[0],
'first_browser_Apple Mail':[0],
'first_browser_Arora':[0],
'first_browser_Avant Browser':[0],
'first_browser_BlackBerry Browser':[0],
'first_browser_Camino':[0],
'first_browser_Chrome':[0],
'first_browser_Chrome Mobile':[0],
'first_browser_Chromium':[0],
'first_browser_CometBird':[0],
'first_browser_Comodo Dragon':[0],
'first_browser_Conkeror':[0],
'first_browser_CoolNovo':[0],
'first_browser_Crazy Browser':[0],
'first_browser_Epic':[0],
'first_browser_Firefox':[0],
'first_browser_Flock':[0],
'first_browser_Google Earth':[0],
'first_browser_Googlebot':[0],
'first_browser_IBrowse':[0],
'first_browser_IE':[0],
'first_browser_IE Mobile':[0],
'first_browser_IceDragon':[0],
'first_browser_IceWeasel':[0],
'first_browser_Iron':[0],
'first_browser_Kindle Browser':[0],
'first_browser_Maxthon':[0],
'first_browser_Mobile Firefox':[0],
'first_browser_Mobile Safari':[0],
'first_browser_Mozilla':[0],
'first_browser_NetNewsWire':[0],
'first_browser_Nintendo Browser':[0],
'first_browser_OmniWeb':[0],
'first_browser_Opera':[0],
'first_browser_Opera Mini':[0],
'first_browser_Opera Mobile':[0],
'first_browser_Outlook 2007':[0],
'first_browser_PS Vita browser':[0],
'first_browser_Pale Moon':[0],
'first_browser_Palm Pre web browser':[0],
'first_browser_RockMelt':[0],
'first_browser_Safari':[0],
'first_browser_SeaMonkey':[0],
'first_browser_Silk':[0],
'first_browser_SiteKiosk':[0],
'first_browser_SlimBrowser':[0],
'first_browser_Sogou Explorer':[0],
'first_browser_Stainless':[0],
'first_browser_TenFourFox':[0],
'first_browser_TheWorld Browser':[0],
'first_browser_UC Browser':[0],
'first_browser_Yandex.Browser':[0],
'first_browser_wOSBrowser':[0]}

#Convert disctionary to a pandas data frame
df = pd.DataFrame.from_dict(input)


#prediction method
def prediction(df):
    prediction = classifier.predict(df)
    #print("User prefered destination:",le_encoder.inverse_transform(np.argsort(prediction)))
    return le_encoder.inverse_transform(np.argsort(prediction))


def main():
    st.title("Destination Prediction")
    html_temp = """
        <div style ="background-color:'';padding:0px">
        <b>Supply inputs to model:</b>
        </div>
        """
    st.markdown(html_temp, unsafe_allow_html=True)
    age = st.text_input("age:", "18")
    input[age]=age
    gender = 'gender_'+st.selectbox ('gender:',('MALE','FEMALE','-unknown-'))
    input[gender] = [1]
    signup_method = 'signup_method'+st.selectbox('signup_method:', ('basic', 'facebook', 'google','weibo'))
    input[signup_method]=[1]
    language = 'language'+st.selectbox('language:', ('ca', 'da', 'en', 'fr','it'))
    input[language]=[1]
    affiliate_channel = 'affiliate_channel'+st.selectbox('affiliate_channel:', ('api', 'content', 'direct', 'remarketing', 'other'))
    input[affiliate_channel]=[1]
    affiliate_provider = 'affiliate_provider'+st.selectbox('affiliate_provider:', ('facebook', 'google', 'yahoo', 'direct', 'craigslist'))
    input[affiliate_provider]=[1]
    first_device_type = 'first_device_type'+st.selectbox('first_device_type:', ('Android Phone', 'Mac Desktop', 'Android Browser', 'Chrome', 'Opera'))
    input[first_device_type]=[1]
    print (input )


    if st.button("Predict"):
        st.success('User prefered destination country: {}'.format(prediction(df)))


if __name__ == '__main__':
    main()