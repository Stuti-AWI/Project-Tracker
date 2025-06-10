# Project Tracker Application

## Overview

**Project Tracker** is a modern, web-based application designed to help research teams, labs, and organizations track samples, experiments, and process workflows. It features robust data management, advanced querying (including natural language and LLM-powered SQL), user authentication, and beautiful, branded UI/UX.

The application is built with Flask, SQLAlchemy, PostgreSQL, and integrates with SendGrid for email notifications. It supports both classic and AI-powered chatbot interfaces for querying and reporting.

---

## Key Features

### 1. Sample and Experiment Management
- **Add, Edit, Delete Samples:** Track detailed information for each sample, including company, ERB, date/time, recipes, glass type, dimensions, process status, and more.
- **Sample Images:** Upload and view sample images (jpg, jpeg, png) with descriptions.
- **Experiment Data:** Attach experiment results (transmittance, reflectance, absorbance, PLQY, SEM, EDX, XRD) to samples.
- **Trash/Restore:** Soft-delete samples and experiments, with the ability to restore from trash.

### 2. Process Workflow Tracking
- **Process Status:** Track cleaning, coating, annealing, and completion status for each sample.
- **Visual Badges:** Statuses are displayed as colored badges (Y/N) in the UI.
- **Automatic "Done" Logic:** "Done" is set to Y only if all process steps are Y.

### 3. Advanced Querying and Chatbot
- **Classic Chatbot:** Query the database using natural language for columns, status, company, ERB, and more.
- **LLM-Powered Chatbot:** Use OpenAI to convert natural language to SQL for advanced queries, including date ranges and custom filters.
- **Date Range Support:** Query by "last month", "last year", "between X and Y", "after X", "before Y", etc.
- **Robust SQL Post-Processing:** Ensures generated SQL is always valid for PostgreSQL, with automatic JOIN correction.

### 4. Data Visualization
- **Plots Page:** Upload and visualize pre- and post-TRA data using Plotly.js.
- **Comparison:** Compare experiment data before and after processing, with statistics and gain calculations.
- **Combined View:** See joined sample and experiment data in a single, filterable table.

### 5. User Authentication and Security
- **Login/Logout:** Secure login with hashed passwords.
- **Registration:** Public registration with duplicate username checks.
- **Forgot/Reset Password:** Users can request password resets via email (SendGrid integration).
- **Admin Controls:** Admin-only actions for certain features.

### 6. Branding and UI/UX
- **Modern Design:** Responsive, clean, and branded interface using Bootstrap 4 and custom CSS.
- **Logo and Branding:** Custom logo, color scheme (#ff1825), and links to company website.
- **Consistent Navigation:** Streamlined navbar with main actions and user controls.

### 7. Prefix Table
- **Manage Prefixes:** Add, view, and delete company/sample prefixes for ID generation.

### 8. Database and Storage
- **PostgreSQL Support:** Uses SQLAlchemy ORM for compatibility with PostgreSQL.
- **MongoDB Integration:** (If enabled) for storing and comparing additional data files.
- **File Uploads:** Stores sample images and experiment data files securely.

### 9. Email Integration
- **SendGrid:** Used for sending password reset emails with branded HTML templates.

---

## Technical Notes

- **Backend:** Flask, SQLAlchemy, PostgreSQL, SendGrid, OpenAI API, python-dotenv
- **Frontend:** Bootstrap 4, Plotly.js, custom CSS, responsive design
- **Security:** Passwords are hashed, sessions are secured, and sensitive keys are loaded from `.env`
- **Extensibility:** Easily add new experiment types, process steps, or chatbot skills

---

## Example User Stories

- **Lab Manager:** Adds new samples, tracks their process status, and uploads experiment results.
- **Researcher:** Uses the chatbot to find all experiments for a specific company in the last month.
- **Admin:** Restores accidentally deleted samples from the trash.
- **User:** Requests a password reset and receives a secure email link.

---

## Getting Started

1. **Install dependencies:**  
   `pip install -r requirements.txt`
2. **Set up environment variables:**  
   Create a `.env` file with your secrets (DB, SendGrid, OpenAI, etc.)
3. **Run the app:**  
   `python app.py`
4. **Access in browser:**  
   `http://localhost:5111` (or your chosen port)

---

## Advanced Query Examples

- "Show all experiments for last month"
- "Show all samples between 01/01/2023 and 03/31/2023"
- "Show all experiments after 01/01/2023"
- "Show all experiments before 12/31/2023"
- "Show id, company name, cleaning, coating columns"
- "Show records where cleaning='Y' and coating='Y'"

---

## Contact & Support

For questions, feature requests, or support, visit [Adaptive Waves](https://www.adaptivewaves.com/). 