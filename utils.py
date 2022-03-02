from colorsys import rgb_to_hls
from joblib import dump, load
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics.pairwise import linear_kernel


def valid_data(sources_and_targets):
    """Returns True if the provided input data can be processed by the classifier.

    :param sources_and_targets: Information about potential source elements and target pages in JSON format
    :type sources_and_targets: dict
    """
    # TODO: Add validation
    if not "sources" in sources_and_targets:
        print("Missing key: 'sources'")
        return False
    sources = sources_and_targets["sources"]
    if not isinstance(sources, list):
        print("Should be list: 'sources'")
        return False
    if not len(sources) > 0:
        print("Should have length greater 0: 'sources'")
        return False
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            print(f"Should be dict: 'sources[{index}]'")
            return False
        if not "id" in source:
            print(f"Missing key: 'id' in 'sources[{index}]'")
            return False
        if not "parentId" in source:
            print(f"Missing key: 'parentId' in 'sources[{index}]'")
            return False
        if not "characters" in source:
            print(f"Missing key: 'characters' in 'sources[{index}]'")
            return False
        if not "color" in source:
            print(f"Missing key: 'color' in 'sources[{index}]'")
            return False
        color = source["color"]
        if not isinstance(color, dict):
            print(f"Should be dict: 'sources[{index}]['color']'")
            return False
        for color_variable in ["r", "g", "b"]:
            if not color_variable in color:
                print(f"Missing key: '{color_variable}' in 'sources[{index}]['color']'")
                return False
            if not isinstance(color[color_variable], float) and not isinstance(
                color[color_variable], int
            ):
                print(f"Should be float: 'sources[{index}]['color'][{color_variable}]'")
                return False
            if not color[color_variable] >= 0:
                print(
                    f"Should be greater or equal 0: 'sources[{index}]['color'][{color_variable}]'"
                )
                return False
            if not color[color_variable] <= 1:
                print(
                    f"Should be less than or equal 1: 'sources[{index}]['color'][{color_variable}]'"
                )
                return False
    if not "targets" in sources_and_targets:
        print("Missing key: 'targets'")
        return False
    targets = sources_and_targets["targets"]
    if not isinstance(targets, list):
        print("Should be list: 'targets'")
        return False
    if not len(targets) > 1:
        print("Should have length greater 1: 'sources'")
        return False
    for index, target in enumerate(targets):
        if not isinstance(target, dict):
            print(f"Should be dict: 'targets[{index}]'")
            return False
        if not "id" in target:
            print(f"Missing key: 'id' in 'sources[{index}]'")
            return False
        if not "topics" in target:
            print(f"Missing key: 'topics' in 'sources[{index}]'")
            return False
        topics = target["topics"]
        if not isinstance(topics, list):
            print(f"Should be list: 'targets[{index}]['topics']'")
            return False
        for topic_index, topic in enumerate(topics):
            if not isinstance(topic, str):
                print(f"Should be str: 'targets[{index}]['topics'][{topic_index}]'")
                return False
    if not "context" in sources_and_targets:
        print("Missing key: 'context'")
        return False
    context = sources_and_targets["context"]
    if not isinstance(context, list):
        print("Should be list: 'context'")
        return False
    return True


def qualifications(sources_and_targets, classifier):
    sources = sources_and_targets["sources"]
    targets = sources_and_targets["targets"]
    context = sources_and_targets["context"]
    penalty_score = len(sources) * len(targets) * -1
    qualifications = []
    for source in sources:
        source_target_scores = []
        for target in targets:
            link_probability = get_link_probability(
                source, target, context, penalty_score, classifier
            )
            source_target_scores.append(link_probability)
        qualifications.append(source_target_scores)
    return {"qualifications": qualifications}


def get_semantic_similarity(source_content, target_topics, application_context):
    # TODO: Add semantic similarity comparison
    # corpus = [source_content, " ".join(target_topics)]
    vectorizer = TfidfVectorizer()
    try:
        all_texts = [" ".join(page_content) for page_content in application_context]
        X = vectorizer.fit_transform(all_texts)
        source_vector = vectorizer.transform([source_content])
        target_vector = vectorizer.transform([" ".join(target_topics)])
        semantic_similarity = linear_kernel(source_vector, target_vector).flatten()[0]
        return semantic_similarity
    except Exception as e:
        # print(e, source_content, target_topics)
        return 0.0


def feature_vector(source, target, application_context):
    # TODO: Add preprocessor
    source_color_rgb = source["color"]
    [hue, lightness, saturation] = rgb_to_hls(
        source_color_rgb["r"], source_color_rgb["g"], source_color_rgb["b"]
    )
    x = [hue, saturation, lightness]
    source_content = source["characters"]
    target_content = target["topics"]
    semantic_similarity = get_semantic_similarity(source_content, target_content, application_context)
    x.append(semantic_similarity)
    return x


def get_link_probability(
    source, target, context, penalty_score, classifier, qualification_threshold=0.5
):
    if source["parentId"] == target["id"]:
        return penalty_score
    sample = feature_vector(source, target, context)
    prediction = classifier.predict_proba([sample])[0]
    link_probability = prediction[1]
    # print(
    #     {
    #         "source": source["name"],
    #         "target": target["name"],
    #         "features": sample,
    #         "proba": link_probability,
    #     }
    # )
    return (
        link_probability
        if link_probability > qualification_threshold
        else link_probability - qualification_threshold
    )


def get_classifier_path(model_id):
    return f"classifiers/{model_id}.joblib"


def create_classifier(model_id):
    clf = SGDClassifier(loss="log")
    clf_path = get_classifier_path(model_id)
    dump(clf, clf_path)


def update_classifier(link_and_label, model_id):
    clf_path = get_classifier_path(model_id)
    clf = load(clf_path)
    is_link = link_and_label["isLink"]
    source = link_and_label["link"]["source"]
    target = link_and_label["link"]["target"]
    context = link_and_label["link"]["context"]
    input_features = feature_vector(source, target, context)
    X = np.array([input_features])
    y = np.array([(int(is_link))])
    clf.partial_fit(X, y, classes=[0, 1])
    dump(clf, clf_path)
