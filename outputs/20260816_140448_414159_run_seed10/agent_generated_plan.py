# Generated from a schema-validated LLM plan; no arbitrary model code.
def execute_plan(controller):
    controller.observe()
    controller.detect('bottle')
    controller.generate_grasps()
    controller.select_grasp()
    controller.execute()
    controller.evaluate()
