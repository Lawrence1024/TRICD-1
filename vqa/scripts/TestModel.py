import json
class TestModel:
    name = ""
    result_dict = {}
    data_dict = {}
    
    def __init__(self, name, data_dict):
        self.name = name
        self.data_dict = data_dict
        
        
    def __str__(self):
        return f"{self.name}"
    
    def train(self) -> bool:
        with open('../tags.json', 'r') as openfile:
            # Reading from json file
            tags_dict = json.load(openfile)
        for tag in tags_dict["questions"]:
            self.result_dict[tag["image_id"]] = 1
        return True
    
    
    def generate_result(self, file_name = None) -> str:
        if not file_name:
            file_name = f"../json_results/{self.name}_response.json"
            
        self.write_result(file_name)
        return file_name
        
    
    def write_result(self, file_name):
        
        json_object = json.dumps(self.result_dict, indent=4)
        with open(file_name, "w+") as outfile:
            outfile.write(json_object)
        
        
        