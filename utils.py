from copy import copy
from joblib import dump, load
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from torch import cosine_similarity
from simplifiedscreen2vec.simplified_embedding import (
    get_figma_component_embedding,
    get_figma_embedding,
    load_bert,
    load_ui_model,
    load_screen_model,
    load_layout_model,
)
from constants import (
    SOURCES,
    TARGETS,
    PAGES,
    ID,
    TYPE,
    BOUNDS,
    X,
    Y,
    WIDTH,
    HEIGHT,
    CHILDREN,
)
from tqdm import tqdm

bert = load_bert()
ui_model = load_ui_model(
    "../Screen2Vec/simplifiedscreen2vec/UI2Vec_model.ep120", bert=bert
)
screen_model = load_screen_model(
    "../Screen2Vec/simplifiedscreen2vec/Screen2Vec_model_v4.ep120"
)
layout_model = load_layout_model(
    "../Screen2Vec/simplifiedscreen2vec/layout_encoder.ep800"
)
clickable_classifier = load("classifiers/clickable.joblib")


def validate_data(pages_data: dict):
    """Raises an error if the provided input data cannot be processed by the classifier.

    :param pages_data: Information about the application's pages in JSON format
    :type pages_data: dict
    """

    def validate_node(node: dict):
        for key in [ID, TYPE, BOUNDS]:
            if not key in node:
                print(node)
                raise Exception(f"Missing key: {key}")
        for key in [X, Y, WIDTH, HEIGHT]:
            if not key in node[BOUNDS]:
                print(node)
                raise Exception(f"Missing key: {key}")
        if CHILDREN in node:
            for child in node[CHILDREN]:
                validate_node(child)

    if not isinstance(pages_data, dict):
        raise Exception(f"Should be dict: pages_data")
    if not PAGES in pages_data:
        raise Exception(f"Missing key: {PAGES}")
    pages = pages_data[PAGES]
    if not isinstance(pages, list):
        raise Exception(f"Should be list: {PAGES}")
    for page_index, page in enumerate(pages):
        if not isinstance(page, dict):
            raise Exception(f"Should be dict: {PAGES}[{page_index}]")
        for key in [ID, WIDTH, HEIGHT, CHILDREN]:
            if not key in page:
                raise Exception(f"Missing key: {key} in {PAGES}[{page_index}]")
        for child in page[CHILDREN]:
            validate_node(child)


def get_element_by_id(element_id, page):
    """
    Returns the element from the specified page tree (DFS search)
    """
    stack = []
    stack.append(page)

    while len(stack) > 0:
        top_element = stack.pop()
        if top_element is not None:
            if top_element["id"] == element_id:
                return top_element
            else:
                if "children" in top_element:
                    for child in top_element["children"]:
                        stack.append(child)


def qualifications(pages_data, classifier, clickable_classifier):
    pages = pages_data["pages"]
    embeddings = [
        get_figma_embedding(page, ui_model, screen_model, layout_model, bert)
        for page in pages
    ]
    targets = [
        {"id": pages[page_index]["id"], "embedding": embedding}
        for page_index, embedding in enumerate(embeddings)
    ]
    sources = [
        {
            "element": {
                "id": component_id,
                "bounds": get_element_by_id(component_id, pages[page_index])["bounds"],
                "embedding": component_embedding,
            },
            "page": {
                "id": pages[page_index]["id"],
                "width": pages[page_index]["width"],
                "height": pages[page_index]["height"],
                "embedding": embedding,
            },
        }
        for page_index, embedding in enumerate(embeddings)
        for component_id, component_embedding in embedding["components"].items()
        if component_id
        != pages[page_index][
            "id"
        ]  # first component is page itself (needs to be excluded)
    ]
    penalty_score = len(sources) * len(targets) * -1
    qualifications = []
    for source in tqdm(sources):
        source_target_scores = []
        for target in targets:
            link_probability, info = get_link_probability(
                source, target, penalty_score, classifier, clickable_classifier
            )
            source_target_scores.append(
                {
                    "source": {"id": source["element"]["id"]},
                    "target": {"id": target["id"]},
                    "probability": link_probability,
                    "info": info,
                }
            )
        qualifications.append(source_target_scores)
    return {"qualifications": qualifications}


def get_element_is_clickable_probability(ui, clickable_classifier):
    """
    Get the probability that a UI element is clickable
    """
    sample = get_ui_element_features(ui)
    prediction = clickable_classifier.predict_proba([sample])[0]
    is_clickable_probability = prediction[1]
    return is_clickable_probability


def get_element_page_text_similarity(ui_embedding, page_text_embedding):
    """
    Get the cosine similarity between the UI embedding and page text embedding vectors
    """
    cos_similarity = cosine_similarity(ui_embedding, page_text_embedding, dim=0)
    similarity_score = cos_similarity.item()
    return similarity_score


def get_link_probability(
    source,
    target,
    penalty_score,
    classifier,
    clickable_classifier,
    qualification_threshold=0.5,
):
    if source["page"]["id"] == target["id"]:
        return penalty_score, {}
    source_is_clickable_probability = get_element_is_clickable_probability(
        source, clickable_classifier
    )
    source_target_text_similarity = get_element_page_text_similarity(
        source["element"]["embedding"], target["embedding"]["text"]
    )
    # target_layout = target["embedding"]["layout"].tolist()
    sample = (
        [source_is_clickable_probability]
        + [source_target_text_similarity]
        # + target_layout
    )
    prediction = classifier.predict_proba([sample])[0]
    link_probability = prediction[1]
    # print(
    #     {
    #         "source": source["id"],
    #         "target": target["id"],
    #         "features": sample,
    #         "proba": link_probability,
    #     }
    # )
    score = (
        link_probability
        if link_probability > qualification_threshold
        else link_probability - qualification_threshold
    )
    return score, {
        "clickableProbability": source_is_clickable_probability,
        "sourceTargetTextSimilarity": source_target_text_similarity,
    }


def get_link_classifier_path(model_id):
    return f"classifiers/{model_id}.joblib"


def get_clickable_classifier_path(model_id):
    return f"classifiers/{model_id}_clickable.joblib"


def create_link_classifier(model_id):
    clf = SGDClassifier(loss="log")
    clf_path = get_link_classifier_path(model_id)
    dump(clf, clf_path)


def create_clickable_classifier(model_id):
    clf = SGDClassifier(loss="log")
    clf_path = get_clickable_classifier_path(model_id)
    dump(clf, clf_path)


def create_classifier(model_id):
    create_clickable_classifier(model_id)
    create_link_classifier(model_id)


def construct_source_dict(source_data):
    source_page_embedding = get_figma_embedding(
        source_data["page"], ui_model, screen_model, layout_model, bert
    )
    source_element_embedding = [
        component_embedding
        for component_id, component_embedding in source_page_embedding[
            "components"
        ].items()
        if component_id == source_data["element"]["id"]
    ][0]
    source = copy(source_data)
    source["element"]["embedding"] = source_element_embedding
    source["page"]["embedding"] = source_page_embedding
    return source


def construct_target_dict(target_data):
    target_embedding = get_figma_embedding(
        target_data, ui_model, screen_model, layout_model, bert
    )
    return {"page": target_data, "embedding": target_embedding}


def update_link_classifier(source, target, is_link, model_id):
    clf_path = get_link_classifier_path(model_id)
    clf = load(clf_path)
    clickable_clf_path = get_clickable_classifier_path(model_id)
    clickable_clf = load(clickable_clf_path)
    source_is_clickable_probability = get_element_is_clickable_probability(
        source, clickable_clf
    )
    source_target_text_similarity = get_element_page_text_similarity(
        source["element"]["embedding"], target["embedding"]["text"]
    )
    X = np.array([[source_is_clickable_probability] + [source_target_text_similarity]])
    y = np.array([(int(is_link))])
    clf.partial_fit(X, y, classes=[0, 1])
    dump(clf, clf_path)


def get_ui_element_features(ui):
    ui_element = ui["element"]
    ui_element_bounds = ui_element["bounds"]
    parent_page = ui["page"]
    parent_page_width = parent_page["width"]
    parent_page_height = parent_page["height"]
    relative_x = ui_element_bounds["x"] / parent_page_width
    relative_y = ui_element_bounds["y"] / parent_page_height
    relative_width = ui_element_bounds["width"] / parent_page_width
    relative_height = ui_element_bounds["height"] / parent_page_height
    features = (
        ui_element["embedding"].tolist()
        + parent_page["embedding"]["screen"].tolist()
        + [relative_x]
        + [relative_y]
        + [relative_width]
        + [relative_height]
    )
    return features


def update_clickable_classifier(source, model_id):
    clf_path = get_clickable_classifier_path(model_id)
    clf = load(clf_path)
    features = get_ui_element_features(source)
    X = np.array([features])
    y = np.array([1])
    clf.partial_fit(X, y, classes=[0, 1])
    dump(clf, clf_path)


def update_classifier(link_and_label, model_id):
    is_link = link_and_label["isLink"]
    source_data = link_and_label["link"]["source"]
    target_data = link_and_label["link"]["target"]
    source = construct_source_dict(source_data)
    target = construct_target_dict(target_data)
    update_clickable_classifier(source, model_id)
    update_link_classifier(source, target, is_link, model_id)
