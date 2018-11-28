while true
do
	nvidia-smi | grep 6078* |grep "% " 
	sleep 1
done
