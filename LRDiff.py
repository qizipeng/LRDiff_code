import torch
from omegaconf import OmegaConf
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, LMSDiscreteScheduler
from my_model import unet_2d_condition
import json
from PIL import Image
from utils import Pharse2idx, draw_box, setup_logger, attentionmap_to_segmentation, compute_ca_loss_new
import hydra
import os
from tqdm import tqdm
import numpy as np
from inversion import DDIMInversion
import copy
from pycocotools.coco import COCO
import imgviz
import cv2
import re  

topk = 10
attention_loss = compute_ca_loss_new

def inference(device, unet, vae, tokenizer, text_encoder, examples_object_list, examples_background, examples_fusion, cfg, logger, lambda1, lambda2, object_stage):
    def get_sigma(device, isstart, time,  now_index, unet, vae, tokenizer, text_encoder, examples, cfg, logger, latents, max_index_step=1, thredhold= 0.9, mask_area = None):
        
        object_positions = Pharse2idx(examples['prompt'], examples['phrases'])

        # Encode Classifier Embeddings
        uncond_input = tokenizer(
            [""] * cfg.inference.batch_size, padding="max_length", max_length=tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
        # Encode Prompt
        input_ids = tokenizer(
            [examples['prompt']] * cfg.inference.batch_size,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        cond_embeddings = text_encoder(input_ids.input_ids.to(device))[0]
        text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])
        if isstart:
            latents = latents * noise_scheduler.init_noise_sigma
        # logger.info("Decode Image...")
        latents_ = 1 / 0.18215 * latents
        init_image = vae.decode(latents_).sample
        init_image = (init_image / 2 + 0.5).clamp(0, 1)   
        loss = torch.tensor(10000)
        attn_map_integrated_downs = []
        attn_map_integrated_mids = []
        attn_map_integrated_ups = []
        for index, t in enumerate(noise_scheduler.timesteps):
            if index <= now_index:
                continue
            with torch.no_grad():
                latent_model_input = torch.cat([latents] * 2)

                latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
                noise_pred, attn_map_integrated_up, attn_map_integrated_mid, attn_map_integrated_down = \
                    unet(latent_model_input, t, encoder_hidden_states=text_embeddings)
                noise_pred = noise_pred.sample
                ### for saving the intermediate attention map
                attn_map_integrated_downs.append(attn_map_integrated_down)
                attn_map_integrated_mids.append(attn_map_integrated_mid)
                attn_map_integrated_ups.append(attn_map_integrated_up)
                if index == now_index + 1:
                    SAVE_WEIGHT = True
                    if SAVE_WEIGHT:
                        segmentation_maps ,segmentation_maps_no_thredhold= attentionmap_to_segmentation(attn_map_integrated_downs,
                                                                            attn_map_integrated_mids,
                                                                            attn_map_integrated_ups, object_positions, 0,
                                                                            1, thredhold, size=512, mask_area = mask_area, isget_postive_color = True)
                        # indices = torch.nonzero(segmentation_maps == 1)
                        top_values, top_indices = torch.topk(segmentation_maps_no_thredhold.flatten(), k=cfg.general.topk, largest=True)
                        select_color_list = [init_image[:,:,idx.item()//512, idx.item()%512] for idx in top_indices]
                        select_color_mean = torch.mean(torch.cat(select_color_list), dim = 0).unsqueeze(0)

                        return select_color_mean , select_color_list
    def update_latents(device, ISstart,  time, now_index, unet, vae, tokenizer, text_encoder, examples_object, cfg, logger, latents, thredhold = 0.9, mask_area = None):
        if ISstart:
            if cfg.general.usingdynamicvector == 10:
                sigma, sigma_list = get_sigma(device, ISstart, time, now_index, unet, vae, tokenizer, text_encoder, examples_object, cfg, logger, latents, thredhold, mask_area= mask_area)
            else:
                sigma, sigma_list = torch.Tensor([[1.0, 1.0, 1.0]]) #[0.3656, 0.4804, 0.3139]
        else:
            sigma = examples_object["sigma"]
            sigma_list = examples_object["sigma_list"]
        with torch.no_grad():
            sigma_repeatdim = sigma.unsqueeze(-1).unsqueeze(-1).repeat(cfg.inference.batch_size,1,8,8).to(device)#torch.ones((cfg.inference.batch_size, 3, 8, 8)).to(device)
            sigma_repeatdim = (sigma_repeatdim - 0.5) * 2.
            sigma_features = vae.encode(sigma_repeatdim).latent_dist.sample().to(device)
            sigma_features = sigma_features *  0.18215
        #the features of the positive area

        if cfg.general.maskinput ==1:
            mask = examples_object["mask"][0][0].unsqueeze(0).unsqueeze(0).repeat(1,4,1,1)

        elif cfg.general.maskinput ==0:
            mask = torch.zeros((1,4,64,64)).float().to(device)
            x1 = int(examples_object['bboxes'][0][0][1]*64)
            x2 = int(examples_object['bboxes'][0][0][3]*64)
            y1 = int(examples_object['bboxes'][0][0][0]*64)
            y2 = int(examples_object['bboxes'][0][0][2]*64)
            mask[:,:,x1:x2,y1:y2] = 1

        latents_update = latents  - lambda1 * sigma_features
        plusvalue = lambda2
        latents_update = latents_update +  mask * (lambda1 + plusvalue) * sigma_features

        return latents_update, sigma, None
    def draw_latents(latents, floder, index, box=None, phrases=None):
        if not os.path.exists(floder):
            os.makedirs(floder)
        latents_ = 1 / 0.18215 * latents
        image = vae.decode(latents_).sample
        image_return = (image / 2 + 0.5).clamp(0, 1)
        image = image_return
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
        images = (image * 255).round().astype("uint8")
        images = Image.fromarray(images)
        image_path = os.path.join(floder,'{}.png'.format(index))
        if box is None:
            images.save(image_path)
            return image_return
        draw_box(images, box, phrases, image_path)
        return image_return
    def img2latent(img):
        latents = vae.encode(img).latent_dist.sample().to(device)
        latents = latents *  0.18215

        return latents

    each_step = [object_stage,object_stage]
    generator = torch.manual_seed(cfg.inference.rand_seed)  # Seed generator to create the inital latent noise
    noise_scheduler = LMSDiscreteScheduler(beta_start=cfg.noise_schedule.beta_start, beta_end=cfg.noise_schedule.beta_end,
                                           beta_schedule=cfg.noise_schedule.beta_schedule, num_train_timesteps=cfg.noise_schedule.num_train_timesteps)

    noise_scheduler.set_timesteps(cfg.inference.timesteps)
    
    latents_init = torch.randn(
        (cfg.inference.batch_size, 4, 64, 64),
        generator=generator,
    ).to(device)

    latents_object_intermediate = []
    segmentation_object_intermediate = []
    object_index = 0
    segmentation_thred = [0.9,0.8,0.7,0.6,0.6,0.6,0.6,0.6]
    examples_index = 0
    for examples_object in examples_object_list:
        examples_index += 1
        logger.info("Inference")

        # Get Object Positions
        logger.info("Conver Phrases to Object Positions: {}".format(examples_object["phrases"]))
        object_positions = Pharse2idx(examples_object['prompt'], examples_object['phrases'])
        # Encode Classifier Embeddings
        uncond_input_bear = tokenizer(
            examples_object["negtive_phrases"] * cfg.inference.batch_size, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
        )
        uncond_embeddings = text_encoder(uncond_input_bear.input_ids.to(device))[0]
        # Encode Prompt
        input_ids_bear = tokenizer(
                [examples_object['prompt']] * cfg.inference.batch_size,
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
        cond_embeddings = text_encoder(input_ids_bear.input_ids.to(device))[0]
        text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])
        
        if cfg.general.maskinput ==1:
            init_sementation_bear = examples_object["mask"][0][0].unsqueeze(0).unsqueeze(0).repeat(1,4,1,1)
        elif cfg.general.maskinput ==0:
            init_sementation_bear = torch.zeros((1,4,64,64)).float().to(device)
            x1 = int(examples_object['bboxes'][0][0][1]*64)
            x2 = int(examples_object['bboxes'][0][0][3]*64)
            y1 = int(examples_object['bboxes'][0][0][0]*64)
            y2 = int(examples_object['bboxes'][0][0][2]*64)
            init_sementation = torch.zeros((1,4,64,64)).float().to(device)
            init_sementation[:,:,x1:x2,y1:y2] = 1.0     
        with torch.no_grad():       
            latents_object, sigma, sigma_list = update_latents(device, True,  noise_scheduler.timesteps[0], -1, unet, vae, tokenizer, text_encoder, examples_object, cfg, logger, latents_init, segmentation_thred[0], mask_area = init_sementation)
        examples_object["sigma"] = sigma
        examples_object["sigma_list"] = sigma_list
        latents_object = latents_object * noise_scheduler.init_noise_sigma

        attn_map_integrated_downs = []
        attn_map_integrated_mids = []
        attn_map_integrated_ups = []
        for index, t in enumerate(tqdm(noise_scheduler.timesteps)):
            iteration = 0
            with torch.no_grad():
                latent_model_input = torch.cat([latents_object] * 2)
                latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
                noise_pred, attn_map_integrated_up, attn_map_integrated_mid, attn_map_integrated_down = \
                    unet(latent_model_input, t, encoder_hidden_states=text_embeddings)
                noise_pred = noise_pred.sample

                ### for saving the intermediate attention map
                attn_map_integrated_downs.append(attn_map_integrated_down)
                attn_map_integrated_mids.append(attn_map_integrated_mid)
                attn_map_integrated_ups.append(attn_map_integrated_up)

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + cfg.inference.classifier_free_guidance * (noise_pred_text - noise_pred_uncond)

                latents_object = noise_scheduler.step(noise_pred, t, latents_object).prev_sample
                draw_latents(latent_model_input[0].unsqueeze(0), os.path.join(cfg.general.save_path, "target_{}".format(examples_index)), index, examples_object["bboxes"], examples_object["phrases"])

                if cfg.general.multiopti:
                    ### new version
                    if index < min(object_stage,5):
                        latents_object = latents_object * ((noise_scheduler.sigmas[index + 1]**2 + 1) ** 0.5)
                        latents_object,_, _ = update_latents(device, False,  t, index, unet, vae, tokenizer, text_encoder,\
                                                                        examples_object, cfg, logger, latents_object, mask_area = init_sementation_bear)
                        latents_object = latents_object / ((noise_scheduler.sigmas[index + 1]**2 + 1) ** 0.5)

                if index == each_step[0]:
                    # latents_img_intermediate_bear = img

                    latents_intermediate = latents_object
                    SAVE_WEIGHT = True
                    if SAVE_WEIGHT:
                        segmentation, segmentation_maps_no_thredhold = attentionmap_to_segmentation(attn_map_integrated_downs,attn_map_integrated_mids, attn_map_integrated_ups, object_positions, 0, index, thredhold = 0.3, size = 64)
                        segmentation = segmentation.unsqueeze(0).unsqueeze(0).repeat(1,4,1,1) * init_sementation_bear
                    break

                torch.cuda.empty_cache()
        latents_object_intermediate.append(latents_intermediate)
        segmentation_object_intermediate.append(segmentation)
        object_index += 1
####-------- background--------####
    logger.info("Conver Phrases to Object Positions: background")
    # object_positions = Pharse2idx(prompt2, phrases)
    # Encode Classifier Embeddings
    uncond_input_background = tokenizer(
        examples_background["negtive_phrases"] * cfg.inference.batch_size, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
    )
    uncond_embeddings_background = text_encoder(uncond_input_background.input_ids.to(device))[0]
    # Encode Prompt
    input_ids_background = tokenizer(
            [examples_background['prompt']] * cfg.inference.batch_size,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )

    cond_embeddings_background = text_encoder(input_ids_background.input_ids.to(device))[0]
    text_embeddings_background = torch.cat([uncond_embeddings_background, cond_embeddings_background])
    latents_background = latents_init * noise_scheduler.init_noise_sigma
    for index, t in enumerate(tqdm(noise_scheduler.timesteps)):
        with torch.no_grad():
            latent_model_input = torch.cat([latents_background] * 2)
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
            noise_pred, attn_map_integrated_up, attn_map_integrated_mid, attn_map_integrated_down = \
                unet(latent_model_input, t, encoder_hidden_states=text_embeddings_background)
            noise_pred = noise_pred.sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg.inference.classifier_free_guidance * (noise_pred_text - noise_pred_uncond)

            latents_background = noise_scheduler.step(noise_pred, t, latents_background).prev_sample
            draw_latents(latent_model_input[0].unsqueeze(0), os.path.join(cfg.general.save_path, "no_target"), index)

            if index == each_step[1]:
                latents_intermediate_background = latents_background
                break

            torch.cuda.empty_cache()

    latents_fusion = latents_intermediate_background#Latents_Fusion(latents_intermediate_background, latents_object_intermediate)
    for i in range(len(latents_object_intermediate)):
            # latents_fusion = latents_fusion +  latents_object_intermediate[i]
            if cfg.general.maskinput==1:
                mask = examples_object_list[i]['mask'][0][0].unsqueeze(0).unsqueeze(0).repeat(1,4,1,1)
            else:
                mask = torch.zeros((1,4,64,64)).float().to(device)
                x1 = int(examples_object_list[i]['bboxes'][0][0][1]*64)
                x2 = int(examples_object_list[i]['bboxes'][0][0][3]*64)
                y1 = int(examples_object_list[i]['bboxes'][0][0][0]*64)
                y2 = int(examples_object_list[i]['bboxes'][0][0][2]*64)
                mask[:,:,x1:x2,y1:y2] = 1
            latents_fusion = latents_fusion * (1-mask) + latents_object_intermediate[i] * mask
    
    logger.info("Conver Phrases to Object Positions: fusion")

    # Encode Classifier Embeddings
    uncond_input_fusion = tokenizer(
        examples_fusion["negtive_phrases"] * cfg.inference.batch_size, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
    )
    uncond_embeddings_fusion = text_encoder(uncond_input_fusion.input_ids.to(device))[0]

    # Encode Prompt
    input_ids_fusion = tokenizer(
            [examples_fusion['prompt']] * cfg.inference.batch_size,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )

    cond_embeddings_fusion = text_encoder(input_ids_fusion.input_ids.to(device))[0]
    text_embeddings_fusion = torch.cat([uncond_embeddings_fusion, cond_embeddings_fusion])
####-------fusion stage-----------####
    for index, t in enumerate(tqdm(noise_scheduler.timesteps)):
        if index<=each_step[1]:
            continue
        with torch.no_grad():
            latent_model_input = torch.cat([latents_fusion] * 2)

            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
            noise_pred, attn_map_integrated_up, attn_map_integrated_mid, attn_map_integrated_down = \
                unet(latent_model_input, t, encoder_hidden_states=text_embeddings_fusion)
            noise_pred = noise_pred.sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg.inference.classifier_free_guidance * (noise_pred_text - noise_pred_uncond)
            latents_fusion = noise_scheduler.step(noise_pred, t, latents_fusion).prev_sample
            draw_latents(latent_model_input[0].unsqueeze(0), os.path.join(cfg.general.save_path, "fusion"), index, examples_fusion["bboxes"], examples_fusion["phrases"])
            torch.cuda.empty_cache()
    with torch.no_grad():
        logger.info("Decode Image...")
        latents_fusion = 1 / 0.18215 * latents_fusion
        image = vae.decode(latents_fusion).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images
    
@hydra.main(version_base=None, config_path="conf", config_name="layered_rendering1")
def main(cfg):
    def myFunc(e):
        if cfg.general.maskinput==1:
            return torch.sum(e["mask"][0][0]) 
        else:
            return (e["bboxes"][0][0][2] - e["bboxes"][0][0][0])  * (e["bboxes"][0][0][3] - e["bboxes"][0][0][1]) 
    if not os.path.exists(cfg.general.save_path):
        os.makedirs(cfg.general.save_path)
    if not os.path.exists(os.path.join(cfg.general.save_path,"results")):
        os.makedirs(os.path.join(cfg.general.save_path,"results"))
    logger = setup_logger(cfg.general.save_path, __name__)
    logger.info(cfg)
    logger.info("save config to {}".format(os.path.join(cfg.general.save_path, 'config.yaml')))
    OmegaConf.save(cfg, os.path.join(cfg.general.save_path, 'config.yaml'))
    with open(cfg.general.unet_config) as f:
        unet_config = json.load(f)
    unet = unet_2d_condition.UNet2DConditionModel(**unet_config).from_pretrained(cfg.general.model_path, subfolder="unet")
    tokenizer = CLIPTokenizer.from_pretrained(cfg.general.model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.general.model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(cfg.general.model_path, subfolder="vae")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet.to(device)
    text_encoder.to(device)
    vae.to(device)

    COCO_annotation = "./data/instances_val2017.json"
    coco_annotation = COCO(annotation_file=COCO_annotation)
    coco_classes1 = dict([(v["id"], v["name"]) for k, v in coco_annotation.cats.items()])
    coco_classes2 = dict([(v["name"], v["id"]) for k, v in coco_annotation.cats.items()])
    
    #### examples:
    # bbox_file = {"bboxes": [{"class": "bear", "bbox": [0.002440273037542662, 0.107515625, 1.0, 0.9886718750000001]}]}
    # caption_file = {"caption": "A big burly grizzly bear is show with grass in the background.", "background": " the grass area."}

    bbox_file = {"bboxes": [{"class": "dog", "bbox": [0.25, 0.2, 0.85, 0.9]}]}
    caption_file = {"caption": "A cute dog is sitting in the garden.", "background": "there is a garden with flowers."}

    lambda1 = cfg.general.lambda1 
    lambda2 = cfg.general.lambda2
    stage = cfg.general.stage

    #### the below code is for processing the captions
    examples_object_list = []
    captions = caption_file["caption"].replace(",","")
    captions = captions.replace(".", "")
    captions = captions.replace("'s", "")
    captions = captions.replace("\n", "")

    for bbox in bbox_file["bboxes"]:
        example = {}
        if cfg.general.usingdynamicvector == 10:
            example["prompt"] = "a " + bbox["class"]                                
        example["phrases"] =  bbox["class"]
        example["bboxes"] = [[bbox["bbox"]]] 
        example["negtive_phrases"] = "",#caption_file["background"],
        example["save_path"] = cfg.general.save_path
        example_mask = None
        example["mask"] = [[example_mask]]
        examples_object_list.append(example)
        captions = captions.replace(bbox["class"]+"s",bbox["class"])
        captions = captions.replace(bbox["class"]+"es",bbox["class"])
        captions = re.sub(r'\b\w*{}\w*\b'.format(bbox["class"]), bbox["class"], captions, flags=re.IGNORECASE)

    examples_background = {
        "prompt": "",
        "phrases": "{}".format(" "),
        "negtive_phrases" : "",
        "bboxes": [[[0,0,0,0]]],
        'save_path': cfg.general.save_path
        }

    examples_background["prompt"] = caption_file["background"]
    examples_object_list.sort(reverse=True, key=lambda k: myFunc(k))
    phrases_fusion = ""
    bboxes_fusion = []
    mask_fusion = []
    i = 0
    for examples_object in examples_object_list:
        examples_object_tmp = copy.deepcopy(examples_object)
        if examples_object["phrases"] in phrases_fusion:
            phrases_tmp = [x.strip() for x in phrases_fusion.split(';')]
            index = phrases_tmp.index(examples_object_tmp["phrases"])
            if cfg.general.maskinput==0:
                tmp_list = copy.deepcopy(examples_object_tmp["bboxes"][0][0])
                bboxes_fusion[index].append(tmp_list)
            else:
                tmp_list = copy.deepcopy(examples_object_tmp["bboxes"][0][0])
                bboxes_fusion[index].append(tmp_list)
                tmp_list = copy.deepcopy(examples_object_tmp["mask"][0][0])
                mask_fusion[index].append(tmp_list)
        else:
            phrases_fusion += examples_object_tmp["phrases"] + ";"
            if cfg.general.maskinput==0:
                tmp_list = copy.deepcopy(examples_object_tmp["bboxes"][0])
                bboxes_fusion.append(tmp_list)
            else:
                tmp_list = copy.deepcopy(examples_object_tmp["bboxes"][0])
                bboxes_fusion.append(tmp_list)
                tmp_list = copy.deepcopy(examples_object_tmp["mask"][0])
                mask_fusion.append(tmp_list)

            if i!= len(examples_object_list) -1:
                examples_background["negtive_phrases"] += examples_object_tmp["phrases"] + "with "
            else:
                examples_background["negtive_phrases"] += examples_object_tmp["phrases"]
    if phrases_fusion[-1] == ";":
        phrases_fusion = phrases_fusion[:-1]

    examples_background["negtive_phrases"] += " ,longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, white area"
    examples_fusion = {
        "prompt": captions,
        "phrases": phrases_fusion,
        "negtive_phrases": "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, white area",
        "bboxes": bboxes_fusion,
        "mask": mask_fusion,
        'save_path': cfg.general.save_path
        }
    
    ### inference
    pil_images = inference(device, unet, vae, tokenizer, text_encoder,examples_object_list, examples_background, examples_fusion, cfg, logger, lambda1, lambda2, object_stage = stage)

    # Save example images
    for index, pil_image in enumerate(pil_images):
        image_path_clear = os.path.join(os.path.join(cfg.general.save_path,"results"), 'result.png')
        image_path = os.path.join(cfg.general.save_path, 'result_box.png')
        pil_image.save(image_path_clear)
        draw_box(pil_image, examples_fusion['bboxes'], examples_fusion['phrases'], image_path)
if __name__ == "__main__":
    main()


# python3 LRDiff.py +general.stage=15 +general.lambda1=0.20 +general.lambda2=0.05  +general.multiopti=1  +general.usingdynamicvector=10  +general.maskinput=0 +general.topk=20 general.save_path='LRDiff' inference.rand_seed=5
# python3 LRDiff.py +general.stage=15 +general.lambda1=0.20 +general.lambda2=0.05  +general.multiopti=1  +general.usingdynamicvector=10  +general.maskinput=0 +general.topk=20 general.save_path='LRDiff' inference.rand_seed=1



# stage : t_0 in the paper
# maskinput : instance mask = 1 / bounding box = 0