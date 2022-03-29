from joblib import dump, load
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
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


def qualifications(pages_data, classifier):
    pages = pages_data["pages"]
    embeddings = [
        get_figma_embedding(page, ui_model, screen_model, layout_model, bert)
        for page in pages
    ]
    targets = [
        {"id": pages[page_index]["id"], "embedding": embedding["screen"].tolist()}
        for page_index, embedding in enumerate(embeddings)
    ]
    sources = [
        {
            "id": component_id,
            "parent_id": pages[page_index]["id"],
            "embedding": component_embedding.tolist(),
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
    for source in sources:
        source_target_scores = []
        for target in targets:
            link_probability = get_link_probability(
                source, target, penalty_score, classifier
            )
            source_target_scores.append(
                {"source": source, "target": target, "probability": link_probability}
            )
        qualifications.append(source_target_scores)
    return {"qualifications": qualifications}


def get_link_probability(
    source, target, penalty_score, classifier, qualification_threshold=0.5
):
    if source["parent_id"] == target["id"]:
        return penalty_score
    sample = source["embedding"] + target["embedding"]
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
    return (
        link_probability
        if link_probability > qualification_threshold
        else link_probability - qualification_threshold
    )


def get_classifier_path(model_id):
    return f"classifiers/{model_id}.joblib"


def create_classifier(model_id):
    clf = SGDClassifier(loss="log", class_weight={0: 0.05, 1: 0.95})
    clf_path = get_classifier_path(model_id)
    dump(clf, clf_path)


def update_classifier(link_and_label, model_id):
    clf_path = get_classifier_path(model_id)
    clf = load(clf_path)
    is_link = link_and_label["isLink"]
    source_page = link_and_label["link"]["source"]["page"]
    source_id = link_and_label["link"]["source"]["element"]["id"]
    target_page = link_and_label["link"]["target"]
    source_page_embedding = get_figma_embedding(
        source_page, ui_model, screen_model, layout_model, bert
    )
    source_embedding = [
        component_embedding
        for component_id, component_embedding in source_page_embedding[
            "components"
        ].items()
        if component_id == source_id
    ][0]
    target_embedding = get_figma_embedding(
        target_page, ui_model, screen_model, layout_model, bert
    )["screen"]
    X = np.array([source_embedding.tolist() + target_embedding.tolist()])
    y = np.array([(int(is_link))])
    clf.partial_fit(X, y, classes=[0, 1])
    dump(clf, clf_path)
