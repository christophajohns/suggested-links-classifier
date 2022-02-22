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
    return True


def qualifications(sources_and_targets):
    sources = sources_and_targets["sources"]
    targets = sources_and_targets["targets"]
    penalty_score = len(sources) * len(targets) * -1
    qualifications = []
    for source in sources:
        source_target_scores = []
        for target in targets:
            link_probability = get_link_probability(source, target, penalty_score)
            source_target_scores.append(link_probability)
        qualifications.append(source_target_scores)
    print(qualifications)
    return {"qualifications": qualifications}


def get_link_probability(source, target, penalty_score):
    print(source, target)
    if source["parentId"] == target["id"]:
        return penalty_score
    return 0.5
