# Running the subsidence app on a Linux virtual machine

What runs: three containers started by one command: the Gradio app (port 7860), Prometheus (9090) and Grafana (3000).
Only port 7860 is open to the outside; the monitoring ports are reachable from the VM itself and through an SSH tunnel.

**VM size (starting point, measure it with `docker stats`):** Ubuntu 22.04 or 24.04, 2 vCPU, 4 GB RAM, 20 GB disk, no GPU.
The first build downloads the Python packages and the Prometheus and Grafana images (several hundred MB), so allow a few minutes.

## 0. Once, on your own computer: put the app in the GitHub repository

The VM gets its code from GitHub, and so will whoever grades it. The public repository
(github.com/Mashouk/subsidence-susceptibility-mapping) does not contain the app yet.

    cd subsidence-susceptibility-mapping
    # copy into the repository root: app.py prepare.py metrics.py feedback.py requirements-app.txt Dockerfile .dockerignore
    #   docker-compose.yml prometheus.yml grafana/ docker-compose.vm.yml .env.example VM_SETUP.md
    printf '\n.env\nartifacts/\nfeedback/\n' >> .gitignore
    git add -A && git status        # check that .env is NOT listed
    git commit -m "Add Gradio app, Docker and monitoring files"
    git push

## 1. On the VM: install Docker and Git

    sudo apt-get update && sudo apt-get install -y git curl ufw
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker $USER      # then log out and back in so the group applies
    docker --version && docker compose version

## 2. Get the code and set the two secrets

    git clone https://github.com/Mashouk/subsidence-susceptibility-mapping.git
    cd subsidence-susceptibility-mapping
    cp .env.example .env
    nano .env                       # set REVIEW_CODE and GRAFANA_PASSWORD (the stack will not start without them)

## 3. Start

    docker compose -f docker-compose.vm.yml up -d --build
    docker compose -f docker-compose.vm.yml ps          # three services "running"
    curl -s localhost:8000/metrics | head                # the app's metrics
    docker compose -f docker-compose.vm.yml logs --tail 20 subsidence-app

The app is at `http://<VM-IP>:7860`. If it does not open from your computer, the cloud provider's firewall (security group)
probably blocks 7860: allow inbound TCP 7860 there. On the VM itself:

    sudo ufw allow OpenSSH && sudo ufw allow 7860/tcp && sudo ufw enable
    ss -tlnp | grep -E '7860|8000|9090|3000'   # 7860 on 0.0.0.0, the others on 127.0.0.1

Docker publishes ports itself and can bypass ufw, which is why the compose file binds the monitoring ports to 127.0.0.1
instead of relying on ufw. Check from outside (your laptop): `curl -m 5 http://<VM-IP>:9090` must fail.

## 4. Look at Prometheus and Grafana from your computer (SSH tunnel)

    ssh -L 3000:localhost:3000 -L 9090:localhost:9090 <user>@<VM-IP>

Keep that window open, then open http://localhost:9090/targets (the target `subsidence-app` must be UP) and
http://localhost:3000 (user `admin`, your GRAFANA_PASSWORD). The dashboard "Subsidence susceptibility app" is provisioned.

## 5. Everyday commands

| Task | Command |
|---|---|
| Update after a `git push` | `git pull && docker compose -f docker-compose.vm.yml up -d --build` |
| Logs | `docker compose -f docker-compose.vm.yml logs -f subsidence-app` |
| Stop (data kept) | `docker compose -f docker-compose.vm.yml down` |
| Stop and delete feedback and metrics history | `docker compose -f docker-compose.vm.yml down -v` |
| Copy the feedback database out | `docker compose -f docker-compose.vm.yml cp subsidence-app:/app/feedback/feedback.db .` |
| CPU and memory of each container | `docker stats --no-stream` |

The containers restart by themselves after a VM reboot (`restart: unless-stopped`).

## 6. Evidence for the Milestone 5 and 6 document

* **8.4 security:** the stack refusing to start without secrets (run `docker compose -f docker-compose.vm.yml config` with no `.env`);
  `ss -tlnp` output; the failed outside request to 9090; the Review tab asking for the review code.
* **8.5 deliberate failure:** `docker compose -f docker-compose.vm.yml stop subsidence-app`, reload the app in the browser (what the user sees),
  look at Prometheus (`up` becomes 0 after the next failed scrape, the interval is 5 s), then `start` it again.
* **9.4 monitoring:** a Grafana screenshot taken after you have used the app, and the same panel after the failure above.
