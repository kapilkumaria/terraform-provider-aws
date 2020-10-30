package aws

import (
	"fmt"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/validation"
)

const (
	// Sagemaker Algorithm BlazingText
	sageMakerRepositoryBlazingText = "blazingtext"
	// Sagemaker Algorithm DeepAR Forecasting
	sageMakerRepositoryDeepARForecasting = "forecasting-deepar"
	// Sagemaker Algorithm Factorization Machines
	sageMakerRepositoryFactorizationMachines = "factorization-machines"
	// Sagemaker Algorithm Image Classification
	sageMakerRepositoryImageClassification = "image-classification"
	// Sagemaker Algorithm IP Insights
	sageMakerRepositoryIPInsights = "ipinsights"
	// Sagemaker Algorithm k-means
	sageMakerRepositoryKMeans = "kmeans"
	// Sagemaker Algorithm k-nearest-neighbor
	sageMakerRepositoryKNearestNeighbor = "knn"
	// Sagemaker Algorithm Latent Dirichlet Allocation
	sageMakerRepositoryLDA = "lda"
	// Sagemaker Algorithm Linear Learner
	sageMakerRepositoryLinearLearner = "linear-learner"
	// Sagemaker Algorithm Neural Topic Model
	sageMakerRepositoryNeuralTopicModel = "ntm"
	// Sagemaker Algorithm Object2Vec
	sageMakerRepositoryObject2Vec = "object2vec"
	// Sagemaker Algorithm Object Detection
	sageMakerRepositoryObjectDetection = "object-detection"
	// Sagemaker Algorithm PCA
	sageMakerRepositoryPCA = "pca"
	// Sagemaker Algorithm Random Cut Forest
	sageMakerRepositoryRandomCutForest = "randomcutforest"
	// Sagemaker Algorithm Semantic Segmentation
	sageMakerRepositorySemanticSegmentation = "semantic-segmentation"
	// Sagemaker Algorithm Seq2Seq
	sageMakerRepositorySeq2Seq = "seq2seq"
	// Sagemaker Algorithm XGBoost
	sageMakerRepositoryXGBoost = "sagemaker-xgboost"
	// Sagemaker Library scikit-learn
	sageMakerRepositoryScikitLearn = "sagemaker-scikit-learn"
	// Sagemaker Library Spark ML
	sageMakerRepositorySparkML = "sagemaker-sparkml-serving"
	// Sagemaker Repo MXNet Inference
	sageMakerRepositoryMXNetInference = "mxnet-inference"
	// Sagemaker Repo MXNet Inference EIA
	sageMakerRepositoryMXNetInferenceEIA = "mxnet-inference-eia"
	// Sagemaker Repo MXNet Training
	sageMakerRepositoryMXNetTraining = "mxnet-training"
	// Sagemaker Repo PyTorch Inference
	sageMakerRepositoryPyTorchInference = "pytorch-inference"
	// Sagemaker Repo PyTorch Inference EIA
	sageMakerRepositoryPyTorchInferenceEIA = "pytorch-inference-eia"
	// Sagemaker Repo PyTorch Training
	sageMakerRepositoryPyTorchTraining = "pytorch-training"
	// Sagemaker Repo TensorFlow Inference
	sageMakerRepositoryTensorFlowInference = "tensorflow-inference"
	// Sagemaker Repo TensorFlow Inference EIA
	sageMakerRepositoryTensorFlowInferenceEIA = "tensorflow-inference-eia"
	// Sagemaker Repo TensorFlow Training
	sageMakerRepositoryTensorFlowTraining = "tensorflow-training"
)

// https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-algo-docker-registry-paths.html
var sageMakerPrebuiltImageIDByRegion_Blazing = map[string]string{
	"ap-east-1":      "286214385809",
	"ap-northeast-1": "501404015308",
	"ap-northeast-2": "306986355934",
	"ap-south-1":     "991648021394",
	"ap-southeast-1": "475088953585",
	"ap-southeast-2": "544295431143",
	"ca-central-1":   "469771592824",
	"cn-north-1":     "390948362332",
	"cn-northwest-1": "387376663083",
	"eu-central-1":   "813361260812",
	"eu-north-1":     "669576153137",
	"eu-west-1":      "685385470294",
	"eu-west-2":      "644912444149",
	"eu-west-3":      "749696950732",
	"me-south-1":     "249704162688",
	"sa-east-1":      "855470959533",
	"us-east-1":      "811284229777",
	"us-east-2":      "825641698319",
	"us-gov-west-1":  "226302683700",
	"us-west-1":      "632365934929",
	"us-west-2":      "433757028032",
}

// https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-algo-docker-registry-paths.html
var sageMakerPrebuiltImageIDByRegion_DeepAR = map[string]string{
	"ap-east-1":      "286214385809",
	"ap-northeast-1": "633353088612",
	"ap-northeast-2": "204372634319",
	"ap-south-1":     "991648021394",
	"ap-southeast-1": "475088953585",
	"ap-southeast-2": "514117268639",
	"ca-central-1":   "469771592824",
	"cn-north-1":     "390948362332",
	"cn-northwest-1": "387376663083",
	"eu-central-1":   "495149712605",
	"eu-north-1":     "669576153137",
	"eu-west-1":      "224300973850",
	"eu-west-2":      "644912444149",
	"eu-west-3":      "749696950732",
	"me-south-1":     "249704162688",
	"sa-east-1":      "855470959533",
	"us-east-1":      "522234722520",
	"us-east-2":      "566113047672",
	"us-gov-west-1":  "226302683700",
	"us-west-1":      "632365934929",
	"us-west-2":      "156387875391",
}

// https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-algo-docker-registry-paths.html
var sageMakerPrebuiltImageIDByRegion_FactorMachines = map[string]string{
	"ap-east-1":      "286214385809",
	"ap-northeast-1": "351501993468",
	"ap-northeast-2": "835164637446",
	"ap-south-1":     "991648021394",
	"ap-southeast-1": "475088953585",
	"ap-southeast-2": "712309505854",
	"ca-central-1":   "469771592824",
	"cn-north-1":     "390948362332",
	"cn-northwest-1": "387376663083",
	"eu-central-1":   "664544806723",
	"eu-north-1":     "669576153137",
	"eu-west-1":      "438346466558",
	"eu-west-2":      "644912444149",
	"eu-west-3":      "749696950732",
	"me-south-1":     "249704162688",
	"sa-east-1":      "855470959533",
	"us-east-1":      "382416733822",
	"us-east-2":      "404615174143",
	"us-gov-west-1":  "226302683700",
	"us-west-1":      "632365934929",
	"us-west-2":      "174872318107",
}

// https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-algo-docker-registry-paths.html
var sageMakerPrebuiltImageIDByRegion_LDA = map[string]string{
	"ap-northeast-1": "258307448986",
	"ap-northeast-2": "293181348795",
	"ap-south-1":     "991648021394",
	"ap-southeast-1": "475088953585",
	"ap-southeast-2": "297031611018",
	"ca-central-1":   "469771592824",
	"eu-central-1":   "353608530281",
	"eu-west-1":      "999678624901",
	"eu-west-2":      "644912444149",
	"us-east-1":      "766337827248",
	"us-east-2":      "999911452149",
	"us-gov-west-1":  "226302683700",
	"us-west-1":      "632365934929",
	"us-west-2":      "266724342769",
}

// https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-algo-docker-registry-paths.html
var sageMakerPrebuiltImageIDByRegion_XGBoost = map[string]string{
	"ap-east-1":      "651117190479",
	"ap-northeast-1": "354813040037",
	"ap-northeast-2": "366743142698",
	"ap-south-1":     "720646828776",
	"ap-southeast-1": "121021644041",
	"ap-southeast-2": "783357654285",
	"ca-central-1":   "341280168497",
	"cn-north-1":     "450853457545",
	"cn-northwest-1": "451049120500",
	"eu-central-1":   "492215442770",
	"eu-north-1":     "662702820516",
	"eu-west-1":      "141502667606",
	"eu-west-2":      "764974769150",
	"eu-west-3":      "659782779980",
	"me-south-1":     "801668240914",
	"sa-east-1":      "737474898029",
	"us-east-1":      "683313688378",
	"us-east-2":      "257758044811",
	"us-gov-west-1":  "414596584902",
	"us-west-1":      "746614075791",
	"us-west-2":      "246618743249",
}

// https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-docker-containers-scikit-learn-spark.html
var sageMakerPrebuiltImageIDByRegion_SparkML = map[string]string{
	"ap-northeast-1": "354813040037",
	"ap-northeast-2": "366743142698",
	"ap-south-1":     "720646828776",
	"ap-southeast-1": "121021644041",
	"ap-southeast-2": "783357654285",
	"ca-central-1":   "341280168497",
	"eu-central-1":   "492215442770",
	"eu-west-1":      "141502667606",
	"eu-west-2":      "764974769150",
	"us-east-1":      "683313688378",
	"us-east-2":      "257758044811",
	"us-gov-west-1":  "414596584902",
	"us-west-1":      "746614075791",
	"us-west-2":      "246618743249",
}

// https://github.com/aws/deep-learning-containers/blob/master/available_images.md
var sageMakerPrebuiltImageIDByRegion_DeepLearning = map[string]string{
	"ap-east-1":      "871362719292",
	"ap-northeast-1": "763104351884",
	"ap-northeast-2": "763104351884",
	"ap-south-1":     "763104351884",
	"ap-southeast-1": "763104351884",
	"ap-southeast-2": "763104351884",
	"ca-central-1":   "763104351884",
	"cn-north-1":     "727897471807",
	"cn-northwest-1": "727897471807",
	"eu-central-1":   "763104351884",
	"eu-north-1":     "763104351884",
	"eu-west-1":      "763104351884",
	"eu-west-2":      "763104351884",
	"eu-west-3":      "763104351884",
	"me-south-1":     "217643126080",
	"sa-east-1":      "763104351884",
	"us-east-1":      "763104351884",
	"us-east-2":      "763104351884",
	"us-west-1":      "763104351884",
	"us-west-2":      "763104351884",
}

func dataSourceAwsSagemakerPrebuiltImagePath() *schema.Resource {
	return &schema.Resource{
		Read: dataSourceAwsRdsOrderableDbInstanceRead,
		Schema: map[string]*schema.Schema{
			"repository_name": {
				Type:     schema.TypeString,
				Required: true,
				ValidateFunc: validation.StringInSlice([]string{
					sageMakerRepositoryBlazingText,
					sageMakerRepositoryDeepARForecasting,
					sageMakerRepositoryFactorizationMachines,
					sageMakerRepositoryImageClassification,
					sageMakerRepositoryIPInsights,
					sageMakerRepositoryKMeans,
					sageMakerRepositoryKNearestNeighbor,
					sageMakerRepositoryLDA,
					sageMakerRepositoryLinearLearner,
					sageMakerRepositoryNeuralTopicModel,
					sageMakerRepositoryObject2Vec,
					sageMakerRepositoryObjectDetection,
					sageMakerRepositoryPCA,
					sageMakerRepositoryRandomCutForest,
					sageMakerRepositorySemanticSegmentation,
					sageMakerRepositorySeq2Seq,
					sageMakerRepositoryXGBoost,
					sageMakerRepositoryScikitLearn,
					sageMakerRepositorySparkML,
					sageMakerRepositoryMXNetInference,
					sageMakerRepositoryMXNetInferenceEIA,
					sageMakerRepositoryMXNetTraining,
					sageMakerRepositoryPyTorchInference,
					sageMakerRepositoryPyTorchInferenceEIA,
					sageMakerRepositoryPyTorchTraining,
					sageMakerRepositoryTensorFlowInference,
					sageMakerRepositoryTensorFlowInferenceEIA,
					sageMakerRepositoryTensorFlowTraining,
				}, false),
			},

			"dns_suffix": {
				Type:     schema.TypeString,
				Optional: true,
			},

			"image_tag": {
				Type:     schema.TypeString,
				Optional: true,
				Default:  "1",
			},

			"region": {
				Type:     schema.TypeString,
				Optional: true,
			},

			"registry_id": {
				Type:     schema.TypeString,
				Computed: true,
			},

			"registry_path": {
				Type:     schema.TypeString,
				Computed: true,
			},
		},
	}
}

func dataSourceAwsSagemakerPrebuiltImagePathRead(d *schema.ResourceData, meta interface{}) error {
	region := meta.(*AWSClient).region
	if v, ok := d.GetOk("region"); ok {
		region = v.(string)
	}

	suffix := meta.(*AWSClient).dnsSuffix
	if v, ok := d.GetOk("dns_suffix"); ok {
		suffix = v.(string)
	}

	repo := d.Get("repository_name").(string)
	imageTag := d.Get("image_tag").(string)

	id := ""
	switch repo {
	case sageMakerRepositoryBlazingText,
		sageMakerRepositoryImageClassification,
		sageMakerRepositoryObjectDetection,
		sageMakerRepositorySemanticSegmentation,
		sageMakerRepositorySeq2Seq:
		id = sageMakerPrebuiltImageIDByRegion_Blazing[region]
	case sageMakerRepositoryDeepARForecasting:
		id = sageMakerPrebuiltImageIDByRegion_DeepAR[region]
	case sageMakerRepositoryLDA:
		id = sageMakerPrebuiltImageIDByRegion_LDA[region]
	case sageMakerRepositoryXGBoost:
		id = sageMakerPrebuiltImageIDByRegion_XGBoost[region]
	case sageMakerRepositoryScikitLearn, sageMakerRepositorySparkML:
		id = sageMakerPrebuiltImageIDByRegion_SparkML[region]
	case sageMakerRepositoryMXNetInference,
		sageMakerRepositoryMXNetInferenceEIA,
		sageMakerRepositoryMXNetTraining,
		sageMakerRepositoryPyTorchInference,
		sageMakerRepositoryPyTorchInferenceEIA,
		sageMakerRepositoryPyTorchTraining,
		sageMakerRepositoryTensorFlowInference,
		sageMakerRepositoryTensorFlowInferenceEIA,
		sageMakerRepositoryTensorFlowTraining:
		id = sageMakerPrebuiltImageIDByRegion_DeepLearning[region]
	default:
		id = sageMakerPrebuiltImageIDByRegion_FactorMachines[region]
	}

	if id == "" {
		return fmt.Errorf("no registry ID available for region (%s) and repository (%s)", region, repo)
	}

	d.SetId(id)
	d.Set("registry_id", id)
	d.Set("registry_path", dataSourceAwsSagemakerPrebuiltImageCreatePath(id, region, suffix, repo, imageTag))
	return nil
}

func dataSourceAwsSagemakerPrebuiltImageCreatePath(id, region, suffix, repo, imageTag string) string {
	return fmt.Sprintf("%s.dkr.ecr.%s.%s/%s:%s", id, region, suffix, repo, imageTag)
}
