"""
Personality Test
Copyright (c) 2022 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 6/20/2022
Updated: 6/21/2022
License: <https://github.com/cannlytics/cannlytics/blob/main/LICENSE>

Disclaimer:

    This test is provided for educational and entertainment uses only.
    The test is not clinically administered and as such the results are
    not suitable for aiding important decisions.
    The test is also fallible, so, if the results say something about
    you that you don't think is true, then you are right and it is wrong.

References:

    - Administering IPIP Measures, with a 50-item Sample Questionnaire
    URL: <https://ipip.ori.org/new_ipip-50-item-scale.htm>

"""
# Standard imports.
import os

# External imports.
from dotenv import dotenv_values
from cannlytics.firebase import initialize_firebase, update_document
from cannlytics.utils import snake_case


PROMPT = """Describe yourself as you generally are now, not as you wish
to be in the future. Describe yourself as you honestly see yourself, in
relation to other people you know of the same sex as you are, and
roughly your same age. So that you can describe yourself in an honest
manner, your responses will be kept in absolute confidence. Indicate for
each statement whether it is

1. Very Inaccurate,
2. Moderately Inaccurate,
3. Neither Accurate Nor Inaccurate,
4. Moderately Accurate, or
5. Very Accurate

as a description of you.
"""

DISCLAIMER = """This test is provided for educational and entertainment uses only.
The test is not clinically administered and as such the results are
not suitable for aiding important decisions.
The test is also fallible, so, if the results say something about
you that you don't think is true, then you are right and it is wrong.
"""

POSITIVE_SCALE = {
    '1': 'Very Inaccurate',
    '2': 'Moderately Inaccurate',
    '3': 'Neither Accurate Nor Inaccurate',
    '4': 'Moderately Accurate',
    '5': 'Very Accurate',
}

NEGATIVE_SCALE = {
    '5': 'Very Inaccurate',
    '4': 'Moderately Inaccurate',
    '3': 'Neither Accurate Nor Inaccurate',
    '2': 'Moderately Accurate',
    '1': 'Very Accurate',
}

FACTORS = {
    '1': 'Extraversion',
    '2': 'Agreeableness',
    '3': 'Conscientiousness',
    '4': 'Neuroticism',
    '5': 'Openness',
}

MAXES = {
    '1': 5 * 5 - (5 * 1),
    '2': 6 * 5 - (4 * 1),
    '3': 6 * 5 - (4 * 1),
    '4': 2 * 5 - (8 * 1),
    '5': 7 * 5 - (3 * 1),
}

MINS = {
    '1': 5 * 1 - (5 * 5),
    '2': 6 * 1 - (4 * 5),
    '3': 6 * 1 - (4 * 5),
    '4': 2 * 1 - (8 * 5),
    '5': 7 * 1 - (3 * 5),
}

QUESTIONS = [
    {'id': '1', 'factor': 1, 'positive': True, 'text': 'Am the life of the party.', 'key': 'EXT1'},
    {'id': '2', 'factor': 2, 'positive': False, 'text': 'Feel little concern for others.', 'key': 'AGR1'},
    {'id': '3', 'factor': 3, 'positive': True, 'text': 'Am always prepared.', 'key': 'CSN1'},
    {'id': '4', 'factor': 4, 'positive': False, 'text': 'Get stressed out easily.', 'key': 'EST1'},
    {'id': '5', 'factor': 5, 'positive': True, 'text': 'Have a rich vocabulary.', 'key': 'OPN1'},
    {'id': '6', 'factor': 1, 'positive': False, 'text': "Don't talk a lot.", 'key': 'EXT2'},
    {'id': '7', 'factor': 2, 'positive': True, 'text': 'Am interested in people.', 'key': 'AGR2'},
    {'id': '8', 'factor': 3, 'positive': False, 'text': 'Leave my belongings around.', 'key': 'CSN2'},
    {'id': '9', 'factor': 4, 'positive': True, 'text': 'Am relaxed most of the time.', 'key': 'EST2'},
    {'id': '10', 'factor': 5, 'positive': False, 'text': 'Have difficulty understanding abstract ideas.', 'key': 'OPN2'},
    {'id': '11', 'factor': 1, 'positive': True, 'text': 'Feel comfortable around people.', 'key': 'EXT3'},
    {'id': '12', 'factor': 2, 'positive': False, 'text': 'Insult people.', 'key': 'AGR3'},
    {'id': '13', 'factor': 3, 'positive': True, 'text': 'Pay attention to details.', 'key': 'CSN3'},
    {'id': '14', 'factor': 4, 'positive': False, 'text': 'Worry about things.', 'key': 'EST3'},
    {'id': '15', 'factor': 5, 'positive': True, 'text': 'Have a vivid imagination.', 'key': 'OPN3'},
    {'id': '16', 'factor': 1, 'positive': False, 'text': 'Keep in the background.', 'key': 'EXT4'},
    {'id': '17', 'factor': 2, 'positive': True, 'text': "Sympathize with others' feelings.", 'key': 'AGR4'},
    {'id': '18', 'factor': 3, 'positive': False, 'text': 'Make a mess of things.', 'key': 'CSN4'},
    {'id': '19', 'factor': 4, 'positive': True, 'text': 'Seldom feel blue.', 'key': 'EST4'},
    {'id': '20', 'factor': 5, 'positive': False, 'text': 'Am not interested in abstract ideas.', 'key': 'OPN4'},
    {'id': '21', 'factor': 1, 'positive': True, 'text': 'Start conversations.', 'key': 'EXT5'},
    {'id': '22', 'factor': 2, 'positive': False, 'text': "Am not interested in other people's problems.", 'key': 'AGR5'},
    {'id': '23', 'factor': 3, 'positive': True, 'text': 'Get chores done right away.', 'key': 'CSN5'},
    {'id': '24', 'factor': 4, 'positive': False, 'text': 'Am easily disturbed.', 'key': 'EST5'},
    {'id': '25', 'factor': 5, 'positive': True, 'text': 'Have excellent ideas.', 'key': 'OPN5'},
    {'id': '26', 'factor': 1, 'positive': False, 'text': 'Have little to say.', 'key': 'EXT6'},
    {'id': '27', 'factor': 2, 'positive': True, 'text': 'Have a soft heart.', 'key': 'AGR6'},
    {'id': '28', 'factor': 3, 'positive': False, 'text': 'Often forget to put things back in their proper place.', 'key': 'CSN6'},
    {'id': '29', 'factor': 4, 'positive': False, 'text': 'Get upset easily.', 'key': 'EST6'},
    {'id': '30', 'factor': 5, 'positive': False, 'text': 'Do not have a good imagination.', 'key': 'OPN6'},
    {'id': '31', 'factor': 1, 'positive': True, 'text': 'Talk to a lot of different people at parties.', 'key': 'EXT7'},
    {'id': '32', 'factor': 2, 'positive': False, 'text': 'Am not really interested in others.', 'key': 'AGR7'},
    {'id': '33', 'factor': 3, 'positive': True, 'text': 'Like order.', 'key': 'CSN7'},
    {'id': '34', 'factor': 4, 'positive': False, 'text': 'Change my mood a lot.', 'key': 'EST7'},
    {'id': '35', 'factor': 5, 'positive': True, 'text': 'Am quick to understand things.', 'key': 'OPN7'},
    {'id': '36', 'factor': 1, 'positive': False, 'text': "Don't like to draw attention to myself.", 'key': 'EXT8'},
    {'id': '37', 'factor': 2, 'positive': True, 'text': 'Take time out for others.', 'key': 'AGR8'},
    {'id': '38', 'factor': 3, 'positive': False, 'text': 'Shirk my duties.', 'key': 'CSN8'},
    {'id': '39', 'factor': 4, 'positive': False, 'text': 'Have frequent mood swings.', 'key': 'EST8'},
    {'id': '40', 'factor': 5, 'positive': True, 'text': 'Use difficult words.', 'key': 'OPN8'},
    {'id': '41', 'factor': 1, 'positive': True, 'text': "Don't mind being the center of attention.", 'key': 'EXT9'},
    {'id': '42', 'factor': 2, 'positive': True, 'text': "Feel others' emotions.", 'key': 'AGR9'},
    {'id': '43', 'factor': 3, 'positive': True, 'text': 'Follow a schedule.', 'key': 'CSN9'},
    {'id': '44', 'factor': 4, 'positive': False, 'text': 'Get irritated easily.', 'key': 'EST9'},
    {'id': '45', 'factor': 5, 'positive': True, 'text': 'Spend time reflecting on things.', 'key': 'OPN9'},
    {'id': '46', 'factor': 1, 'positive': False, 'text': 'Am quiet around strangers.', 'key': 'EXT10'},
    {'id': '47', 'factor': 2, 'positive': True, 'text': 'Make people feel at ease.', 'key': 'AGR10'},
    {'id': '48', 'factor': 3, 'positive': True, 'text': 'Am exacting in my work.', 'key': 'CSN10'},
    {'id': '49', 'factor': 4, 'positive': False, 'text': 'Often feel blue.', 'key': 'EST10'},
    {'id': '50', 'factor': 5, 'positive': True, 'text': 'Am full of ideas.', 'key': 'OPN10'},
]


def score_personality_test(data):
    """Score a personality test for the "Big 5" personality traits.
    Normalizes the scores from the range of possible scores.
    """
    scores, normalized = {}, {}
    for factor in FACTORS: scores[factor] = 0
    for question in QUESTIONS:
        _id = question['id']
        value = int(data[_id])
        factor = str(question['factor'])
        if question['positive']:
            scores[factor] += value
        else:
            scores[factor] -= value
    for factor, value in scores.items():
        maximum = MAXES[factor]
        minimum = MINS[factor]
        normalized[factor] = (value - minimum) / (maximum - minimum)
    return {snake_case(FACTORS[k]):round(normalized[k], 2) for k in normalized}


if __name__ == '__main__':

    # Test the personality test.
    print('Testing personality test....')
    test = {
        '1': 3,
        '2': 3,
        '3': 3,
        '4': 3,
        '5': 3,
        '6': 3,
        '7': 3,
        '8': 3,
        '9': 3,
        '10': 3,
        '11': 3,
        '12': 3,
        '13': 3,
        '14': 3,
        '15': 3,
        '16': 3,
        '17': 3,
        '18': 3,
        '19': 3,
        '20': 3,
        '21': 3,
        '22': 3,
        '23': 3,
        '24': 3,
        '25': 3,
        '26': 3,
        '27': 3,
        '28': 3,
        '29': 3,
        '30': 3,
        '31': 3,
        '32': 3,
        '33': 3,
        '34': 3,
        '35': 3,
        '36': 3,
        '37': 3,
        '38': 3,
        '39': 3,
        '40': 3,
        '41': 3,
        '42': 3,
        '43': 3,
        '44': 3,
        '45': 3,
        '46': 3,
        '47': 3,
        '48': 3,
        '49': 3,
        '50': 3,
    }
    score = score_personality_test(test)
    for factor, value in score.items():
        assert value == 0.5
    print('Neutral score works.')

    # Initialize Firebase
    config = dotenv_values('../../.env')
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config['GOOGLE_APPLICATION_CREDENTIALS']
    db = initialize_firebase()

    # Upload the constants to Firestore.
    constants = {
        'prompt': PROMPT,
        'disclaimer': DISCLAIMER,
        'factors': FACTORS,
        'maxes': MAXES,
        'mins': MINS,
        'positive_scale': POSITIVE_SCALE,
        'negative_scale': NEGATIVE_SCALE,
        'questions': QUESTIONS,
    }
    ref = 'public/data/variables/personality_test'
    update_document(ref, constants, database=db)
 