from open_vocab_grasping.pipeline import detection_image_size, detection_prompts


def test_detection_prompt_ensemble_is_configured_and_deduplicated() -> None:
    config = {"detection": {"text_prompts": {"mug": ["mug", "cup", "mug"]}}}
    assert detection_prompts(config, " Mug ") == ["mug", "cup"]
    assert detection_prompts(config, "bottle") == ["bottle"]


def test_detection_image_size_can_be_target_specific() -> None:
    config = {"detection": {"image_size": 640, "image_size_by_target": {"bowl": 960}}}
    assert detection_image_size(config, "bowl") == 960
    assert detection_image_size(config, "mug") == 640
