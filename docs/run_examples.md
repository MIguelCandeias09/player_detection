https://github.com/roboflow/sports


macOS
First setup:
Paste "sports" root folder onto examples/soccer

pyenv global 3.11.11



Windows 11:
Open terminal in Admin Mode:
conda activate cuda


MODES:
	PITCH_DETECTION
	PLAYER_DETECTION
	BALL_DETECTION
	PLAYER_TRACKING
	TEAM_CLASSIFICATION
	RADAR 



change radar size in /sports/annotators/soocer.py -> scale = 0.1 default (big)


RUN ALL files in folder

python main.py --source_video_path videos/lp1/round2/

python main.py --source_video_path videos/lp1/round1/ --target_video_path videos/lp1/round2/out_pitch --mode PITCH_DETECTION

python main.py --source_video_path videos/lp1/round1/ --target_video_path videos/lp1/round2/out_pitch_640_v11m --mode PITCH_DETECTION

python main.py --source_video_path videos/lp1/round1/ --target_video_path videos/lp1/round2/out_radar_640_v11m --mode RADAR

python main.py --source_video_path videos/lp1/round1/J1-Sporting-RioAve_3-0.mp4 --target_video_path=test/test.mp4 --mode RADAR

python main.py --source_video_path videos/lp1/round2/ --mode RADAR

python main.py --source_video_path videos/lp1/round2/ --mode RADAR --device mps










Run file by file


python main.py --source_video_path videos/braga_vs_moreirense_2_1.mp4 --target_video_path videos/braga_vs_moreirense_2_1_tracking.mp4 --device=cpu --mode=TEAM_CLASSIFICATION

python main.py --source_video_path videos/braga_vs_moreirense_2_1_720p.mp4 --target_video_path videos/braga_vs_moreirense_2_1_720p_tracking_RADAR.mp4 --device=cpu --mode=RADAR

python main.py --source_video_path videos/braga_vs_moreirense_2_1_720p.mp4 --target_video_path videos/braga_vs_moreirense_2_1_720p_tracking_PITCH.mp4 --device=cpu --mode=PITCH_DETECTION

python main.py --source_video_path videos/braga_vs_moreirense_2_1_720p.mp4 --target_video_path videos/braga_vs_moreirense_2_1_720p_tracking_TEAM.mp4 --device=cpu --mode=TEAM_CLASSIFICATION

python main.py --source_video_path videos/braga_vs_moreirense_2_1_720p.mp4 --target_video_path videos/braga_vs_moreirense_2_1_720p_tracking_BALL.mp4 --device=cpu --mode=BALL_DETECTION

python main.py --source_video_path videos/braga_vs_moreirense_2_1_720p.mp4 --target_video_path videos/braga_vs_moreirense_2_1_720p_tracking_PLAYER_D.mp4 --device=cpu --mode=PLAYER_DETECTION

python main.py --source_video_path videos/braga_vs_moreirense_2_1_720p.mp4 --target_video_path videos/braga_vs_moreirense_2_1_720p_tracking_PLAYER_T.mp4 --device=cpu --mode=PLAYER_TRACKING

python main.py --source_video_path videos/2024-08-31-STA_AVS_golo_1-1_cut.mp4 --target_video_path videos/2024-08-31-STA_AVS_golo_1-1_radar.mp4 --mode=RADAR

python main.py --source_video_path videos/2024-09-01_VSC_FAM_1-1_1.mp4 --target_video_path videos/2024-09-01_VSC_FAM_1-1_1_radar.mp4 --mode=RADAR
python main.py --source_video_path videos/2024-09-01_VSC_FAM_1-1_2.mp4 --target_video_path videos/2024-09-01_VSC_FAM_1-1_2_radar.mp4 --mode=RADAR

python main.py --source_video_path videos/2024-09-01_VSC_FAM_1-1_2.mp4 --target_video_path videos/2024-09-01_VSC_FAM_1-1_2_pitch.mp4 --mode=PITCH_DETECTION

python main.py --source_video_path videos/2024-09-01-AMA_CPI_0-1.mp4 --target_video_path videos/2024-09-01-AMA_CPI_0-1.mp4_radar.mp4 --mode=RADAR

python main.py --source_video_path videos/2024-08-24_POR_RAV_1-0.mp4 --target_video_path videos/2024-08-24_POR_RAV_1-0_radar.mp4 --mode=RADAR

python main.py --source_video_path videos/2024-08-24_POR_RAV_1-1.mp4 --target_video_path videos/2024-08-24_POR_RAV_1-1_radar.mp4 --mode=RADAR

python main.py --source_video_path videos/2024-08-25_AVS_VSC_1-0_1.mp4 --target_video_path videos/2024-08-25_AVS_VSC_1-0_1_radar.mp4 --mode=RADAR

python main.py --source_video_path videos/2024-08-25_AVS_VSC_1-0_2.mp4 --target_video_path videos/2024-08-25_AVS_VSC_1-0_2_radar.mp4 --mode=RADAR

python main.py --source_video_path videos/2024-08-25_ARO_NAC_1-0.mp4 --target_video_path videos/2024-08-25_ARO_NAC_1-0_radar.mp4 --mode=RADAR

python main.py --source_video_path videos/2024-08-24_SLB_AMA_1-0.mp4 --target_video_path videos/2024-08-24_SLB_AMA_1-0_radar.mp4 --mode=RADAR

python main.py --source_video_path videos/2024-08-24_CPI_STA_0-1.mp4 --target_video_path videos/2024-08-24_CPI_STA_0-1_radar.mp4 --mode=RADAR

python main.py --source_video_path videos/2024-08-24_CPI_STA_0-2.mp4 --target_video_path videos/2024-08-24_CPI_STA_0-2_radar.mp4 --mode=RADAR

python main.py --source_video_path videos/2024-08-23-FAR-SPO_0-1.mp4 --target_video_path videos/2024-08-23-FAR-SPO_0-1_radar.mp4 --mode=RADAR
python main.py --source_video_path videos/2024-08-23-FAR-SPO_0-3.mp4 --target_video_path videos/2024-08-23-FAR-SPO_0-3_radar.mp4 --mode=RADAR
python main.py --source_video_path videos/2024-08-23-FAR-SPO_0-4.mp4 --target_video_path videos/2024-08-23-FAR-SPO_0-4_radar.mp4 --mode=RADAR
python main.py --source_video_path videos/2024-08-23-FAR-SPO_0-5.mp4 --target_video_path videos/2024-08-23-FAR-SPO_0-5_radar.mp4 --mode=RADAR

python main.py --source_video_path videos/2024-08-24-FAM-BOA_1-0.mp4 --target_video_path videos/2024-08-24-FAM-BOA_1-0_radar.mp4 --mode=RADAR

python main.py --source_video_path videos/2024-08-19_AMA-FAM_0-1.mp4 --target_video_path videos/2024-08-19_AMA-FAM_0-1_radar.mp4 --mode=RADAR
python main.py --source_video_path videos/2024-08-19_AMA-FAM_0-2.mp4 --target_video_path videos/2024-08-19_AMA-FAM_0-2_radar.mp4 --mode=RADAR
python main.py --source_video_path videos/2024-08-19_AMA-FAM_0-3.mp4 --target_video_path videos/2024-08-19_AMA-FAM_0-3_radar.mp4 --mode=RADAR

python main.py --source_video_path videos/EUR24_FINAL_SPA-ENG_1-0.mp4 --target_video_path videos/EUR24_FINAL_SPA-ENG_1-0_radar.mp4 --mode=RADAR

python main.py --source_video_path videos/EUR24_SEMI_SPA-FRA_0-1.mp4 --target_video_path videos/EUR24_SEMI_SPA-FRA_0-1_radar.mp4 --mode=RADAR
python main.py --source_video_path videos/EUR24_SEMI_SPA-FRA_1-1.mp4 --target_video_path videos/EUR24_SEMI_SPA-FRA_1-1_radar.mp4 --mode=RADAR
python main.py --source_video_path videos/EUR24_SEMI_SPA-FRA_2-1.mp4 --target_video_path videos/EUR24_SEMI_SPA-FRA_2-1_radar.mp4 --mode=RADAR

python main.py --source_video_path videos/EUR24_SEMI_NED-ENG_1-0.mp4 --target_video_path videos/EUR24_SEMI_NED-ENG_1-0_radar.mp4 --mode=RADAR

python main.py --source_video_path videos/EUR24_SEMI_NED-ENG_1-2.mp4 --target_video_path videos/EUR24_SEMI_NED-ENG_1-2_radar.mp4 --mode=RADAR

python main.py --source_video_path videos/EUR24_SEMI_SPA-GER_1-0.mp4 --target_video_path videos/EUR24_SEMI_SPA-GER_1-0_radar.mp4 --mode=RADAR
python main.py --source_video_path videos/EUR24_SEMI_SPA-GER_1-1.mp4 --target_video_path videos/EUR24_SEMI_SPA-GER_1-1_radar.mp4 --mode=RADAR
python main.py --source_video_path videos/EUR24_SEMI_SPA-GER_2-1.mp4 --target_video_path videos/EUR24_SEMI_SPA-GER_2-1_radar.mp4 --mode=RADAR

python main.py --source_video_path videos/EUR24_SEMI_ENG-SWI_0-1.mp4 --target_video_path videos/EUR24_SEMI_ENG-SWI_0-1_radar.mp4 --mode=RADAR

python main.py --source_video_path videos/EUR24_SEMI_ENG-SWI_1-1.mp4 --target_video_path videos/EUR24_SEMI_ENG-SWI_1-1_radar.mp4 --mode=RADAR

python main.py --source_video_path videos/EUR24_SEMI_NED-TUR_0-1.mp4 --target_video_path videos/EUR24_SEMI_NED-TUR_0-1_radar.mp4 --mode=RADAR --device=cuda
python main.py --source_video_path videos/EUR24_SEMI_NED-TUR_1-1.mp4 --target_video_path videos/EUR24_SEMI_NED-TUR_1-1_radar.mp4 --mode=RADAR
python main.py --source_video_path videos/EUR24_SEMI_NED-TUR_2-1.mp4 --target_video_path videos/EUR24_SEMI_NED-TUR_2-1_radar.mp4 --mode=RADAR

python main.py --source_video_path videos/EUR24_SEMI_NED-TUR_0-1_720.mp4 --target_video_path videos/EUR24_SEMI_NED-TUR_0-1_radar.mp4 --mode=RADAR --device=cuda

python main.py --source_video_path videos/EUR24_SEMI_NED-TUR_0-1_480.mp4 --target_video_path videos/EUR24_SEMI_NED-TUR_0-1_radar.mp4 --mode=RADAR --device=cuda




#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################


python main.py --source_video_path videos/lp1/round1/J7-Braga-RioAve_1-0.mp4 --target_video_path videos/lp1/round1/out/J7-Braga-RioAve_1-0_radar.mp4 --mode=RADAR --device=cuda

python main.py --source_video_path videos/lp1/round1/J7-Braga-RioAve_1-0.mp4 --target_video_path videos/lp1/round1/out/J7-Braga-RioAve_1-0_player.mp4 --mode=PLAYER_DETECTION --device=cuda

python main.py --source_video_path videos/lp1/round1/J7-EstrelaAmadora-Moreirense_1-0.mp4 --target_video_path videos/lp1/round1/out/J7-EstrelaAmadora-Moreirense_1-0_player.mp4 --mode=PLAYER_DETECTION --device=cuda

python main.py --source_video_path videos/lp1/round1/J11-Benfica-Porto_2-1.mp4 --target_video_path videos/lp1/round1/out/J11-Benfica-Porto_2-1_player.mp4 --mode=PLAYER_DETECTION --device=cuda
J11-Benfica-Porto_2-1

python main.py --source_video_path videos/lp1/round1/J3-Braga-Moreirense_1-0.mp4 --target_video_path videos/lp1/round1/out/J3-Braga-Moreirense_1-0_player.mp4 --mode=PLAYER_DETECTION --device=cuda

python main.py --source_video_path videos/lp1/round1/J11-Estrela-Nacional_2-0.mp4 --target_video_path videos/lp1/round1/out/J11-Estrela-Nacional_2-0_player.mp4 --mode=PLAYER_DETECTION --device=cuda

python main.py --source_video_path videos/lp1/round1/J12-Porto-CasaPia_1-0.mp4 --target_video_path videos/lp1/round1/out/J12-Porto-CasaPia_1-0_player.mp4 --mode=PLAYER_DETECTION --device=cuda

python main.py --source_video_path videos/lp1/round1/J12-Sporting-SantaClara_0-1.mp4 --target_video_path videos/lp1/round1/out/J12-Sporting-SantaClara_0-1_player.mp4 --mode=PLAYER_DETECTION --device=cuda

python main.py --source_video_path videos/lp1/round1/J12-Sporting-SantaClara_0-1.mp4 --target_video_path videos/lp1/round1/out/J12-Sporting-SantaClara_0-1_player.mp4 --mode=TEAM_CLASSIFICATION --device=cuda

python main.py --source_video_path videos/lp1/round1/J2-Boavista-Braga_0-1.mp4 --target_video_path videos/lp1/round1/out/J2-Boavista-Braga_0-1_player.mp4 --mode=PLAYER_DETECTION --device=cuda

python main.py --source_video_path videos/lp1/round1/J2-Boavista-Braga_0-1.mp4 --target_video_path videos/lp1/round1/out/J2-Boavista-Braga_0-1_team.mp4 --mode=TEAM_CLASSIFICATION --device=cuda

python main.py --source_video_path videos/lp1/round1/J1-Sporting-RioAve_3-1.mp4 --target_video_path videos/lp1/round1/out/J1-Sporting-RioAve_3-1_radar.mp4 --mode=RADAR --device=cuda
