"""
Subjective Effects and Aromas Prediction Model
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 5/13/2022
Updated: 5/18/2022
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>

Description:

    A re-usable model that predicts potential aromas and effects
    given lab results for strains, products, etc.

Data Sources:

    - Data from: Over eight hundred cannabis strains characterized
    by the relationship between their subjective effects, perceptual
    profiles, and chemical compositions
    URL: <https://data.mendeley.com/datasets/6zwcgrttkp/1>
    License: CC BY 4.0. <https://creativecommons.org/licenses/by/4.0/>

Resources:

    - Over eight hundred cannabis strains characterized by the
    relationship between their psychoactive effects, perceptual
    profiles, and chemical compositions
    URL: <https://www.biorxiv.org/content/10.1101/759696v1.abstract>

    - Effects of cannabidiol in cannabis flower:
    Implications for harm reduction
    URL: <https://pubmed.ncbi.nlm.nih.gov/34467598/>

"""
# Standard imports.
from datetime import datetime
import os

# External imports.
from cannlytics.firebase import (
    download_file,
    get_document,
    initialize_firebase,
    update_documents,
    upload_file,
)
from cannlytics.utils import kebab_case, snake_case
from cannlytics.utils.utils import nonzero_columns, nonzero_rows
from cannlytics.utils import download_file_from_url, unzip_files
import pandas as pd
# from sklearn.metrics import confusion_matrix
import statsmodels.api as sm


# Decarboxylation rate. Source: <https://www.conflabs.com/why-0-877/>
DECARB = 0.877


def download_strain_review_data(
        data_dir,
        url='https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/6zwcgrttkp-1.zip',
):
    """Download historic strain review data.
    First, creates the data directory if it doesn't already exist.
    Second, downloads the data to the given directory.
    Third, unzips the data and returns the directories.
    Source: "Data from: Over eight hundred cannabis strains characterized
    by the relationship between their subjective effects, perceptual
    profiles, and chemical compositions".
    URL: <https://data.mendeley.com/datasets/6zwcgrttkp/1>
    License: CC BY 4.0. <https://creativecommons.org/licenses/by/4.0/>
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    download_file_from_url(url, destination=data_dir)
    unzip_files(data_dir)
    # Optional: Get the directories programmatically.
    strain_folder = 'Strain data/strains'
    compound_folder = 'Terpene and Cannabinoid data'
    return {'strains': strain_folder, 'compounds': compound_folder}


def curate_lab_results(
        data_dir,
        compound_folder='Terpene and Cannabinoid data',
        cannabinoid_file='rawDATACana',
        terpene_file='rawDATATerp',
        max_cannabinoids=35,
        max_terpenes=8,
):
    """Curate lab results for effects prediction model."""

    # Read terpenes.
    terpenes = None
    if terpene_file:
        file_path = os.path.join(data_dir, compound_folder, terpene_file)
        terpenes = pd.read_csv(file_path, index_col=0)
        terpenes.columns = [snake_case(x).strip('x_') for x in terpenes.columns]
        terpene_names = list(terpenes.columns[3:])
        compounds = terpenes

    # Read cannabinoids.
    cannabinoids = None
    if cannabinoid_file:
        file_path = os.path.join(data_dir, compound_folder, cannabinoid_file)
        cannabinoids = pd.read_csv(file_path, index_col=0)
        cannabinoids.columns = [snake_case(x).strip('x_') for x in cannabinoids.columns]
        cannabinoid_names = list(cannabinoids.columns[3:])
        compounds = cannabinoids

    # Merge terpenes and cannabinoids.
    if terpene_file and cannabinoid_file:
        compounds = pd.merge(
            left=cannabinoids,
            right=terpenes,
            left_on='file',
            right_on='file',
            how='left',
            suffixes=['', '_terpene']
        )

    # Rename any oddly named columns.
    rename = {
        'cb_da': 'cbda',
        'cb_ga': 'cbda',
        'delta_9_th_ca': 'delta_9_thca',
        'th_ca': 'thca',
    }
    compounds.rename(columns=rename, inplace=True)

    # Combine `delta_9_thca` and `thca`.
    # FIXME: Ensure that this is combining the two fields correctly.
    compounds['delta_9_thca'].fillna(compounds['thca'], inplace=True)
    compounds.drop(columns=['thca'], inplace=True)
    cannabinoid_names.remove('thca')

    # FIXME: Calculate totals.
    compounds['total_terpenes'] = compounds[terpene_names].sum(axis=1).round(2)
    compounds['total_cannabinoids'] = compounds[cannabinoid_names].sum(axis=1).round(2)
    compounds['total_thc'] = (compounds['delta_9_thc'] + compounds['delta_9_thca'].mul(DECARB)).round(2)
    compounds['total_cbd'] = (compounds['cbd'] + compounds['cbda'].mul(DECARB)).round(2)

    # Exclude outliers.
    compounds = compounds.loc[
        (compounds['total_cannabinoids'] < max_cannabinoids) &
        (compounds['total_terpenes'] < max_terpenes)
    ]

    # Return average results by strain,
    # counting the number of lab results per strain.
    concentrations = compounds.groupby('tag').mean()
    concentrations = concentrations.fillna(0)
    concentrations['tests'] = compounds.groupby('tag')['cbd'].count()
    return concentrations


def curate_strain_reviews(
        data_dir, 
        results,
        strain_folder='Strain data/strains',
):
    """Curate cannabis strain reviews."""

    # Create a panel of reviews of strain lab results.
    panel = pd.DataFrame()
    for _, row in results.iterrows():

        # Read the strain's effects and aromas data.
        review_file = row.name + '.p'
        file_path = os.path.join(data_dir, strain_folder, review_file)
        try:
            strain = pd.read_pickle(file_path)
        except FileNotFoundError:
            print("Couldn't find:", row.name)
            continue

        # Assign dummy variables for effects and aromas.
        reviews = strain['data_strain']
        name = strain['strain']
        category = list(strain['categorias'])[0]
        for n, review in enumerate(reviews):

            # Create panel observation, combining prior compound data.
            obs = row.copy()
            for aroma in review['sabores']:
                key = 'aroma_' + snake_case(aroma)
                obs[key] = 1
            for effect in review['efectos']:
                key = 'effect_' + snake_case(effect)
                obs[key] = 1

            # Assign category determined from original authors NLP.
            obs['category'] = category
            obs['strain'] = name

            # Record the observation.
            obs.name = name + '-' + str(n)
            obs = obs.to_frame().transpose()
            panel = pd.concat([panel, obs])

        # Optional: Estimate the probability of 
        # a review containing aroma or effect.
        # This was the original authors strategy.

        # Optional: NLP of 'reporte'.
        # This is how original author classified hybrid, indica, and sativa.

        # Optional: Assign dummy variables for mode aromas.

    # Return the panel with null effects and aromas as 0.
    panel = panel.fillna(0)
    return panel


def estimate_effects_model(X, Y, method=None):
    """Estimate a potential effects prediction model.
    The algorithm excludes all null columns, adds a constant,
    then fits probit model(s) for each effect variable.
    The user can specify their model, e.g. logit, probit, etc.
    """
    X = X.loc[:, (X != 0).any(axis=0)]
    X = sm.add_constant(X)
    models = {}
    if method == 'logit':
        method = sm.Logit
    elif method is None or method == 'probit':
        method = sm.Probit
    for variable in Y.columns:
        try:
            y = Y[variable]
            model = method(y, X).fit()
            models[variable] = model
        except:
            models[variable] = None # Error estimating!
    return models


def calculate_model_statistics(models, Y, X):
    """Determine prediction thresholds for a given model."""
    X = X.loc[:, (X != 0).any(axis=0)]
    x = sm.add_constant(X)
    thresholds = {}
    for key in Y.columns:
        y = Y[key]
        y_bar = y.mean()
        model = models[key]
        if model:
            x_hat = x[list(model.params.keys())]
            y_hat = model.predict(x_hat)
            threshold = y_hat.quantile(1 - y_bar)
            thresholds[key] = threshold

        # TODO: Calculate the confusion matrix and
        # also return summary statistics.
        # if Y:
        #     cm = confusion_matrix(Y, predictions)
        #     tn, fp, fn, tp = cm.ravel()
            # Strains predicted correctly with the aroma / effect:
            # actual = list(panel.loc[panel[independent_variable] == 1].index.unique())
            # predicted = list(predictions.loc[predictions == 1].index.unique())
            # correct = list(set(actual) & set(predicted))
            # accuracy = round(len(correct) / len(actual) * 100, 2)
            # print('Strain classification accuracy: %.2f%%' % accuracy)
    return {'thresholds': thresholds}


def predict_effects(models, X, thresholds):
    """Predict potential aromas and effects for a cannabis product(s)
    given. Add a constant column if necessary and only use model columns."""
    x = X.assign(const=1)
    predictions = pd.DataFrame()
    for key, model in models.items():
        if not model:
            predictions[key] = 0
            continue
        x_hat = x[list(model.params.keys())]
        y_hat = model.predict(x_hat)
        threshold = thresholds[key]
        prediction = pd.Series(y_hat > threshold).astype(int)
        predictions[key] = prediction
    return predictions


def upload_effects_model(model, ref, name=None, data_dir='/tmp', stats=None):
    """Upload a prediction model for future use.
    Pickle model, upload the file to Firebase Storage,
    and record the file's data in Firebase Firestore.
    """
    # TODO: Zip all the pickle files?
    if name is None:
        name = kebab_case(ref)
    model_path = os.path.join(data_dir, f'{name}.pickle')
    model.save(model_path)
    upload_file(ref, model_path)
    if stats:
        data = stats
    else:
        data = {}
    data['model_ref'] = ref
    update_documents(ref, data)
    return data


def get_effects_model(ref, data_dir='/tmp'):
    """Get a pre-built prediction model for use.
    First, gets the model data from Firebase Firestore.
    Second, downloads the pickle file and loads it into a model.
    """
    name = kebab_case(ref)
    data = get_document(ref)
    pickle_file = os.path.join(data_dir, f'{name}.pickle')
    download_file(ref, pickle_file)
    data['model'] = sm.load(pickle_file)
    return data


def save_predictions(model, data):
    """Save a model's predictions."""
    refs = [f'public/data/strains/{x}' for x in data.index]
    entries = [{'potential_effects': x[0]} for x in data.values]
    
    # TODO: Save accuracy statistics with potential effects and aromas.
    # data['model_params'] = model.params

    # Optional: Parse out effects / aromas at this stage?

    timestamp = datetime.now().isoformat()[:19].replace(':', '-')
    logs = [f'models/effects/predictions/{timestamp}-{x}' for x in data.index]
    update_documents(refs, entries)
    # update_documents(logs, entries)


def download_dataset(name, destination):
    """Download a Cannlytics dataset by its name and given a destination."""
    short_url = f'https://cannlytics.page.link/{name}'
    download_file_from_url(short_url, destination=destination)


#-----------------------------------------------------------------------

if __name__ == '__main__':

    #-------------------------------------------------------------------
    # Fit the model with the training data.
    #-------------------------------------------------------------------

    print('Testing...')
    DATA_DIR = '../.datasets/subjective-effects'

    # Initialize Firebase.
    # db = initialize_firebase(env_file='../../../.env')

    # Optional: Download the original data!
    # download_strain_review_data(DATA_DIR)

    # Curate the lab results.
    strain_data = curate_lab_results(DATA_DIR)

    # Curate the reviews.
    reviews = curate_strain_reviews(DATA_DIR, strain_data)

    # Save the reviews.
    reviews.to_excel(DATA_DIR + '/strain-reviews.xlsx')

    # Read back in the reviews.
    reviews = pd.read_excel(DATA_DIR + '/strain-reviews.xlsx', index_col=0)

    # Add category to strain data.
    strain_data['strain'] = strain_data.index
    strain_data = pd.merge(
        strain_data,
        reviews[['strain', 'category']],
        how='left',
        left_on='strain',
        right_on='strain',
    )
    strain_data = strain_data.groupby('strain').first()

    # Future work: Upload individual lab results for each strain.
    # Format the lab results as metrics with CAS, etc.
    # refs = [f'public/data/strains/{x[0]}/lab_results/{x[1]}' for x in ]

    # Upload the strain data to Firestore.
    refs = [f'public/data/strains/{x}' for x in strain_data.index]
    data = strain_data.to_dict(orient='records')
    # update_documents(refs, data, database=db)

    # Upload strain review data to Firestore.
    reviews['id'] = reviews.index
    refs = [f'public/data/strains/{x[0]}/strain_reviews/{x[1]}' for x in reviews[['strain', 'id']].values]
    data = reviews.to_dict(orient='records')
    # update_documents(refs, data, database=db)

    # Optional: Upload the datasets to Firebase Storage for easy access.

    # Optional: Download the pre-compiled data from Cannlytics.
    # strain_data = download_dataset('strains', DATA_DIR)
    # reviews = download_dataset('strain-reviews', DATA_DIR)

    # Future work: Estimate different types of prediction models:
        # - Logit
        # - Terpene only
        # - Cannabinoid only
        # - Total cannabinoid model: Total CBD, total THC, total minor
        # - Top 1-3 terpenes
        # - Top 1-3 terpenes + total cannabinoids
        # - Compare with random forest?
        # - Use terpene ratios!!!
    
    # Do it Bayesian!!!

    # Use the data to create an effect prediction model.
    exclude = ['total_cannabinoids', 'total_terpenes', 'total_thc',
               'total_cbd', 'count', 'tests', 'category']
    aromas = [x for x in reviews.columns if x.startswith('aroma')]
    effects = [x for x in reviews.columns if x.startswith('effect')]
    variates = [x for x in strain_data.columns if x not in exclude]
    Y = reviews[aromas + effects]
    X = reviews[variates]
    models = estimate_effects_model(X, Y)

    # Calculate statistics for the model.
    model_stats = calculate_model_statistics(models, Y, X)

    # Use the model for prediction!
    predictions = predict_effects(models, X, model_stats['thresholds'])
    predicted_effects = predictions.apply(nonzero_rows, axis=1)

    # Save the predictions!
    strain_effects = predicted_effects.to_frame()
    strain_effects['strain'] = reviews['strain']
    strain_effects = strain_effects.groupby('strain').first()
    # save_predictions(model_stats, strain_effects)

    # Save the model.
    # upload_effects_model(models, 'models/effects/full', stats=model_stats)


    #-------------------------------------------------------------------
    # Demonstrate use in the wild.
    #-------------------------------------------------------------------

    # Get the model.
    # models = get_effects_model('models/effects/full')

    # Predict a single sample (below are mean concentrations).
    strain_name = 'Test Sample'
    x = pd.DataFrame([{
        'delta_9_thc': 10.85,
        'cbd': 0.29,
        'cbn': 0.06,
        'cbg': 0.54,
        'cbc': 0.15,
        'thcv': 0.07,
        'cbda': 0.40,
        'delta_8_thc': 0.00,
        'cbga': 0.40,
        'delta_9_thca': 8.64,
        'd_limonene': 0.22,
        'beta_ocimene': 0.05,
        'beta_myrcene': 0.35,
        'beta_pinene': 0.12,
        'linalool': 0.07,
        'alpha_pinene': 0.10,
        'caryophyllene': 0.18,
        'camphene': 0.01,
        '3_carene': 0.00,
        'alpha_terpinene': 0.00,
        'ocimene': 0.00,
        'cymene': 0.00,
        'eucalyptol': 0.00,
        'gamma_terpinene': 0.00,
        'terpinolene': 0.80,
        'isopulegol': 0.00,
        'geraniol': 0.00,
        'humulene': 0.06,
        'transnerolidol_1': 0.00,
        'transnerolidol_2': 0.01,
        'guaiol': 0.01,
        'caryophylleneoxide': 0.00,
        'alpha_bisabolol': 0.03,
        'beta_caryophyllene': 0.11,
        'alpha_humulene': 0.03,
        'pcymene': 0.00,
        'p_cymene': 0.00,
        'trans_nerolidol': 0.00,
        'terpinene': 0.00,
    }])
    prediction = predict_effects(models, x, model_stats['thresholds'])
    effects = nonzero_columns(prediction)
    print(f'Predicted effects and aromas for {strain_name}:', effects)

    # Save / log the prediction.
    # data = {
    #     'potential_effects': effects,
    #     'lab_results': x.to_dict(),
    #     'model': 'full',
    # }
    # ref = f'public/strains/{kebab_case(strain_name)}'
    # update_document(ref, data)
